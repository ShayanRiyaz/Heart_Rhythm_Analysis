import numpy as np
import matplotlib.pyplot as plt

class MSPTDFastV2BeatDetector:
    """
    Python port of Peter Charlton's MSPTDfast v2 beat detector,
    matching the MATLAB `msptdfastv2_beat_detector` functionality.
    """

    def __init__(
        self,
        min_scale: int = 1,
        max_scale: int | None = None,
        use_reduced_lms_scales: bool = False,
        do_ds: bool = False,
        ds_freq: float = np.nan,
        win_len: float = 8.0,
        win_overlap: float = 0.2,
        optimisation: bool = False,
        plaus_hr_bpm: tuple[float, float] = (30.0, 200.0),
        find_peaks: bool = True,
        find_onsets: bool = True
    ):
        # MATLAB defaults:
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.use_reduced_lms_scales = use_reduced_lms_scales
        self.do_ds = do_ds
        self.ds_freq = ds_freq
        self.win_len = win_len
        self.win_overlap = win_overlap
        self.optimisation = optimisation
        self.plaus_hr_bpm = plaus_hr_bpm
        self.find_peaks = find_peaks
        self.find_onsets = find_onsets

    def detect(self, sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        # x = sig.ravel()
        # N = x.size
        x = np.asarray(sig).ravel()
        N = x.size      
        # Window settings (MATLAB uses round for hop)
        win_samples = int(self.win_len * fs)
        if win_samples < 1:
            raise ValueError("Window length too small for sampling rate")
        hop = int(round(win_samples * (1 - self.win_overlap))) or 1

        peaks_all = []
        onsets_all = []
        start = 0
        while start < N:
            end = min(start + win_samples, N)
            sig_win = x[start:end]
            p_win, t_win = self._process_window(sig_win, fs)
            peaks_all.extend(p_win + start)
            onsets_all.extend(t_win + start)
            if end == N:
                break
            start += hop

        peaks = self._tidy_beats(np.array(peaks_all, dtype=int))
        onsets = self._tidy_beats(np.array(onsets_all, dtype=int))
        return peaks, onsets

    def _process_window(self, sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        # -- downsample if requested (MATLAB rounds ds_factor) --
        if self.do_ds and not np.isnan(self.ds_freq) and fs > self.ds_freq:
            ds_factor = int(round(fs / self.ds_freq))
            ds_fs = fs / ds_factor
            x_ds = sig[::ds_factor]
        else:
            ds_factor = 1
            ds_fs = fs
            x_ds = sig
        N_ds = x_ds.size

        # -- determine max scale on downsampled signal --
        if self.max_scale is None:
            max_scale_ds = int(np.floor(N_ds / 2)) - 1
        else:
            max_scale_ds = int(np.floor(self.max_scale / ds_factor))

        # -- build full-scale masks --
        m_max_full, m_min_full = self._build_scale_masks(x_ds, max_scale_ds)
        scales = np.arange(self.min_scale, max_scale_ds + 1)

        # -- prune by physiological HR range (MATLAB only enforces lower bound) --
        if self.use_reduced_lms_scales:
            hr_bpm = 30.0 * ds_fs / scales
            keep = hr_bpm >= self.plaus_hr_bpm[0]
            if not np.any(keep):
                return np.array([], int), np.array([], int)
            rows = np.where(keep)[0]
            m_max = m_max_full[rows, :]
            m_min = m_min_full[rows, :]
        else:
            m_max = m_max_full
            m_min = m_min_full

        # -- optional scale optimisation via inverse-gamma (prune scales only) --
        if self.optimisation and self.find_peaks and m_max.size:
            gamma_max = np.sum(m_max, axis=1)
            k_opt = int(np.argmin(1.0 / (gamma_max+1e-5))) + 1
            m_max = m_max[:k_opt, :]
        if self.optimisation and self.find_onsets and m_min.size:
            gamma_min = np.sum(m_min, axis=1)
            k_opt_min = int(np.argmin(1.0 / (gamma_max+1e-5))) + 1
            m_min = m_min[:k_opt_min, :]

        # -- coarse detection --
        if self.find_peaks:
            votes_max = m_max.shape[0] - np.sum(m_max, axis=0)
            coarse_p = np.flatnonzero(votes_max == 0)
            peaks = self._refine_extrema(coarse_p, sig, ds_factor, mode="max")
        else:
            peaks = np.array([], int)

        if self.find_onsets:
            votes_min = m_min.shape[0] - np.sum(m_min, axis=0)
            coarse_t = np.flatnonzero(votes_min == 0)
            onsets = self._refine_extrema(coarse_t, sig, ds_factor, mode="min")
        else:
            onsets = np.array([], int)

        return peaks, onsets

    def _build_scale_masks(self, x: np.ndarray, max_scale: int) -> tuple[np.ndarray, np.ndarray]:
        # Vectorized neighbor-difference masks (MATLAB Method 2 / 6)
        N = x.size
        scales = np.arange(self.min_scale, max_scale + 1)
        K = scales.size
        indices = np.arange(N)
        idx_lo = np.clip(indices - scales[:, None], 0, N - 1)
        idx_hi = np.clip(indices + scales[:, None], 0, N - 1)
        S1 = x[None, :]
        S2 = x[idx_lo]
        S3 = x[idx_hi]
        m_max = (S1 > S2) & (S1 > S3)
        m_min = (S1 < S2) & (S1 < S3)
        return m_max, m_min

    def _refine_extrema(self, coarse_idxs: np.ndarray, x: np.ndarray, ds: int, mode: str) -> np.ndarray:
        if coarse_idxs.size == 0:
            return coarse_idxs
        refined = []
        N = x.size
        for i in coarse_idxs:
            center = int(i * ds)
            lo = max(center - ds, 0)
            hi = min(center + ds, N - 1)
            segment = x[lo:hi + 1]
            offset = np.argmax(segment) if mode == "max" else np.argmin(segment)
            refined.append(lo + offset)
        return np.array(sorted(set(refined)), dtype=int)

    def _tidy_beats(self, beats: np.ndarray) -> np.ndarray:
        # MATLAB `tidy_beats`: sort and de-duplicate only
        if beats.size == 0:
            return beats
        unique_beats = np.unique(np.sort(beats))
        return unique_beats

# Example usage (synthetic PPG)
# if __name__ == "__main__":
#     fs = 100.0
#     t = np.linspace(0, 10, int(fs*10))
#     ppg = 0.5 + 0.4 * np.sin(2 * np.pi * 1.2 * t) + 0.05 * np.random.randn(len(t))

#     detector = MSPTDFastV2BeatDetector(
#         use_reduced_lms_scales=True,
#         do_ds=True,
#         ds_freq=50.0,
#         optimisation=False
#     )
#     peaks, onsets = detector.detect(ppg, fs)
#     print("Detected peaks indices:", peaks)
#     print("Detected onsets indices:", onsets)

#     plt.figure()
#     plt.plot(t, ppg)
#     plt.plot(peaks / fs, ppg[peaks], 'o')
#     plt.plot(onsets / fs, ppg[onsets], 'x')
#     plt.xlabel("Time (s)")
#     plt.ylabel("PPG amplitude")
#     plt.title("PPG with detected peaks (o) and onsets (x)")
#     plt.show()
