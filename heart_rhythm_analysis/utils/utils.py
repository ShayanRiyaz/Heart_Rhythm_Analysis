import torch
from heart_rhythm_analysis.peak_detectors.mmpvd2 import MSPTDFastV2BeatDetector
import numpy as np 
from torch.utils.data import Dataset
import h5py
from scipy.signal import firwin, filtfilt, lfilter

from fractions import Fraction
import numpy as np
from scipy.signal import firwin, filtfilt, lfilter, resample, resample_poly

def decimate_signal(x: np.ndarray,
                    fs_in: float,
                    fs_out: float,
                    cutoff: list[float,float] = None,
                    numtaps: int = 31,
                    zero_phase: bool = True) -> np.ndarray:
    """
    Down-sample a 1-D signal from fs_in to fs_out.
    - If fs_in/fs_out is an integer, does FIR anti-alias + take every Nth sample.
    - Otherwise, uses polyphase resampling for fractional ratios.

    Parameters
    ----------
    x         : 1-D numpy array, the raw signal.
    fs_in     : original sampling rate (Hz).
    fs_out    : desired output rate (Hz).
    cutoff    : low-pass cutoff (Hz); defaults to 0.8 * (fs_out) to avoid aliasing.
    numtaps   : number of taps in the FIR filter (odd recommended).
    zero_phase: if True use filtfilt; else causal lfilter.

    Returns
    -------
    y         : down-sampled 1-D array at ~fs_out.
    """
    if x.ndim != 1:
        raise ValueError("Input must be 1-D")
    # default cutoff just below new Nyquist
    if cutoff is None:
        cutoff = (fs_out / 2)
    # compute ideal ratio
    ratio = fs_in / fs_out
    # integer decimation?
    if abs(ratio - round(ratio)) < 1e-6:
        factor = int(round(ratio))
        # design anti-alias FIR at original rate
        # numtaps = 31  # e.g., keep it short relative to signal window
        # cutoff = [0.3, 8]  # bandpass range in Hz
        if len(x) < 3 * numtaps:
            raise ValueError(f"Signal too short ({len(x)} samples) for {numtaps}-tap filter")

        h = firwin(numtaps, cutoff, pass_zero=False, fs=fs_in)
        filt = filtfilt if zero_phase else lfilter
        x_filt = filt(h, 1.0, x)
        return x_filt[::factor]
    # fractional decimation via polyphase
    # compute up/down from the desired ratio
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator

    out_signal = resample_poly(x, up, down)
    return out_signal

# ------------------------------------------------------------------
# 1.  Utility that cleans one 1‑D numpy array -----------------------
# ------------------------------------------------------------------
def clean_signal(arr: np.ndarray) -> np.ndarray | None:
    if arr.ndim != 1:
        raise ValueError("signal must be 1-D")
    ok = ~np.isnan(arr)
    if not ok.any():
        return None
    first, last = ok.argmax(), len(arr) - ok[::-1].argmax()
    arr = arr[first:last].astype("float32")
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        idx = np.arange(arr.size)
        arr[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], arr[~nan_mask])
    return arr


def find_sliding_window(input_size, target_windows, overlap):
    """
    Finds window_size, hop_size, and number of windows closest to target_windows.

    If `overlap == 0` and `input_size` divides evenly by `target_windows`, returns
    the exact division `(input_size//target_windows, input_size//target_windows, target_windows)`.

    Args:
        input_size (int): Total number of samples in your signal.
        target_windows (int): Desired number of windows.
        overlap (int): Number of samples overlapping between consecutive windows.

    Returns:
        tuple: (window_size, hop_size, num_windows), or None if no valid configuration.
    """
    # Exact division when no overlap
    if overlap == 0 and input_size % target_windows == 0:
        win = input_size // target_windows
        return (win, win, target_windows)

    best_config = None
    best_diff = float('inf')

    # Search for best approximate configuration
    for window_size in range(overlap + 1, input_size + 1):
        hop_size = window_size - overlap
        if hop_size <= 0:
            continue

        num_windows = (input_size - overlap) // hop_size
        if num_windows < 1:
            continue

        diff = abs(num_windows - target_windows)
        divides_evenly = (input_size - overlap) % hop_size == 0
        best_divides = False
        if best_config is not None:
            best_divides = ((input_size - overlap) % best_config[1] == 0)

        # Prefer smaller diff, then even division
        if diff < best_diff or (diff == best_diff and divides_evenly and not best_divides):
            best_diff = diff
            best_config = (window_size, hop_size, num_windows)

    return best_config

from typing import Callable, Tuple, Optional

def scale_signal(
    input_vector: np.ndarray,
    config: Tuple[int, int, int] = None,
    window_fn: Optional[Callable[[int], np.ndarray]] = None,
    method: str = 'norm'
) -> np.ndarray:
    """
    Applies standardization or normalization over sliding windows defined by config,
    and reconstructs the full signal via overlap-add.

    Args:
        input_vector (np.ndarray): 1D array of shape (N,).
        config (tuple): (window_size, hop_size, _num_windows_ignored).
        window_fn (callable, optional): fn(length) -> 1D weights array.
                                         Defaults to flat (ones).
        method (str): 'standard' for z-score, 'norm' for min-max then zero-mean.

    Returns:
        np.ndarray: scaled array of same shape as input_vector.
    """
    if input_vector.ndim != 1:
        raise ValueError("input_vector must be 1-D")

    N = input_vector.shape[0]
    if config is None:
        window_size = hop_size = N
    else:
        window_size,hop_size, _ = config

    # default to flat window
    if window_fn is None:
        window_fn = lambda L: np.ones(L, dtype=input_vector.dtype)

    out = np.zeros(N, dtype=input_vector.dtype)
    weights = np.zeros(N, dtype=input_vector.dtype)

    for start in range(0, N, hop_size):
        end = min(start + window_size, N)
        seg = input_vector[start:end]

        # compute z-scores or min-max then zero-mean
        if method == 'standard':
            mu = seg.mean()
            sigma = seg.std(ddof=0)
            if sigma == 0:
                z = np.zeros_like(seg)
            else:
                z = (seg - mu) / sigma

        elif method == 'norm':
            mn, mx = seg.min(), seg.max()
            denom = mx - mn
            if denom == 0:
                z = np.zeros_like(seg)
            else:
                z = (seg - mn) / denom
                z = z - z.mean()  # zero-center after scaling

        else:
            raise ValueError(f"Unknown method '{method}'")

        # get window weights
        w = window_fn(end - start)
        w = np.asarray(w, dtype=input_vector.dtype)
        if w.ndim != 1 or w.shape[0] != end - start:
            raise ValueError("window_fn must return a 1-D array of length (end-start)")

        # accumulate
        out[start:end] += z * w
        weights[start:end] += w

        if end == N:
            break

    # normalize by total weight
    mask = weights > 0
    out[mask] /= weights[mask]

    return out

def pseudo_peak_vector(seg, fs=20.83):
    """Return soft (Gaussian) peak map for a 1‑D segment."""

    detector = MSPTDFastV2BeatDetector(
        use_reduced_lms_scales=True,
        do_ds=True,
        ds_freq=fs,
        optimisation=True
    )       
    _, locs = detector.detect(seg, fs)

    y = np.zeros_like(seg, dtype='float32')
    y[locs] = 1.
    return y

class PPGWindow(Dataset):
    def __init__(self, h5_path, win_len: int = 625, transform=None):
        """
        h5_path: path to your windowed HDF5
        win_len: the length you want every window to be
        transform: optional callable applied to proc_ppg Tensor
        """
        
        self.h5_path = h5_path
        self.win_len = win_len
        self.h5 = h5py.File(h5_path, 'r', swmr=True)
        self.transform = transform

        # build flat index
        self.index = [
            (subj, wid)
            for subj in self.h5.keys()
            for wid in self.h5[subj].keys()
        ]

    def __len__(self):
        return len(self.index)

    def _pad_or_trim(self, arr: np.ndarray):
        L = arr.shape[0]
        if L < self.win_len:
            # wrap-pad if you like, or constant-pad with zero
            return np.pad(arr, (0, self.win_len - L), mode='wrap')
        else:
            return arr[:self.win_len]

    def __getitem__(self, idx):
        subj, wid = self.index[idx]
        grp = self.h5[subj][wid]

        # print(self.h5.keys())
        # print(self.index[idx])
        # print(self.h5[subj].keys())
        # print(grp.keys())
        # print(grp.items())
        # print(grp.attrs['raw_ppg_fs'])
        # raw arrays from file
        proc = grp['proc_ppg'][:]    # float32, length maybe ≠ win_len
        y    = grp['y'][:]           # float32, same
        raw  = grp['raw_ppg'][:]     # float64, maybe full trace
        

        # enforce uniform length
        proc = self._pad_or_trim(proc).astype('float16')
        y    = self._pad_or_trim(y).astype('float16')
        # raw  = self._pad_or_trim(raw).astype('float32')

        # to torch
        proc_t = torch.from_numpy(proc).float().unsqueeze(0)  # (1, win_len)
        y_t    = torch.from_numpy(y).float()                 # (win_len,)
        raw_t  = torch.from_numpy(raw).float().unsqueeze(0)  # (1, win_len)
        # raw_fs  = torch.tensor(raw_fs, dtype=torch.float16)

        if self.transform:
            proc_t = self.transform(proc_t)

        label = grp.attrs.get('label')
        raw_fs = grp.attrs.get('raw_ppg_fs')
        return proc_t, y_t, raw_t, label,raw_fs

    def __getstate__(self):
        st = self.__dict__.copy()
        st.pop('h5', None)
        return st

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.h5 = h5py.File(self.h5_path, 'r', swmr=True)