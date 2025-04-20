from __future__ import annotations

"""MMPDV2 Beat Detector – Python implementation of the Mountaineer's Method (v2).

This module provides a :class:`MMPDV2BeatDetector` that replicates the MATLAB
`mmpdv2_beat_detector` wrapper and the core `Alpinista_simple_4_todos` peak
finder described in *Prada 2019*.

References
----------
E. J. A. Prada, "The mountaineer's method for peak detection in
photoplethysmographic signals," *Revista Facultad de Ingenieria*, 90, 42–50,
2019. https://doi.org/10.17533/udea.redin.n90a06

License
-------
MIT – see original MATLAB notice. Port by ChatGPT, 2025-04-18.
"""

from typing import Tuple, Sequence, List

import numpy as np

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = ["MMPDV2BeatDetector"]


class MMPDV2BeatDetector:
    """Detect pulse peaks and onsets in a PPG signal.

    Parameters
    ----------
    refractory : float, default 0.35
        Initial refractory period *in seconds*. Increase (≈0.65) for
        non‑paediatric recordings as recommended in the original code.
    tidy_min_distance : float, default 0.25
        *Seconds* below which adjacent peaks are considered duplicates and the
        smaller one is discarded in :pymeth:`_tidy_beats`.
    """

    def __init__(
        self,
        *,
        refractory: float = 0.35,
        tidy_min_distance: float = 0.25,
    ) -> None:
        self.refractory0 = float(refractory)
        self.tidy_min_distance = float(tidy_min_distance)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def detect(self, sig: Sequence[float] | np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Run the detector.

        Parameters
        ----------
        sig : 1‑D array‑like
            Photoplethysmogram samples.
        fs : float
            Sampling frequency *(Hz)*.

        Returns
        -------
        peaks : ndarray[int]
            Indices of detected pulse *peaks*.
        onsets : ndarray[int]
            Indices of detected pulse *onsets* (valleys preceding each peak).
        """
        sig = np.asarray(sig, dtype=float).ravel()
        if sig.ndim != 1:
            raise ValueError("sig must be 1‑D")
        if fs <= 0:
            raise ValueError("fs must be positive")

        peaks = self._alpinista_simple(sig, fs)
        peaks = self._tidy_beats(peaks, fs)
        peaks = self._refine_peaks(sig, peaks, fs)
        onsets = self._pulse_onsets_from_peaks(sig, peaks)
        return peaks, onsets

    # ------------------------------------------------------------------
    # Core algorithm – direct port of Alpinista_simple_4_todos
    # ------------------------------------------------------------------
    def _alpinista_simple(self, sig: np.ndarray, fs: float) -> np.ndarray:
        """Return *candidate* peak indices via Mountaineer's Method v2.

        This is a verbatim, index‑safe Python port of the MATLAB routine
        ``Alpinista_simple_4_todos``. Only the outputs required by the wrapper
        (``t_pico`` a.k.a. *peaks*) are computed; the many fiducial features
        are omitted for clarity but can be re‑enabled easily.
        """
        ascend_max = int(round(0.6 * fs * 0.15))
        ascend_cnt = 0
        first_peak_found = False

        ppi_value = 1.0  # seconds
        refractory = float(self.refractory0)

        peaks: List[int] = []  # t_pico in samples (0‑based)
        valleys: List[int] = []  # t_min in samples (0‑based)

        # Pre‑allocate ascent counter etc. if one wants to debug visually
        # (not returned to user, hence skipped).

        N = sig.size
        for i in range(1, N):  # 0‑based; MATLAB started at 2
            if sig[i] > sig[i - 1]:
                ascend_cnt += 1
                continue

            # Descent started – check if the preceding ascent is large enough
            if ascend_cnt >= ascend_max:
                peak_idx = i - 1
                valley_idx = i - (ascend_cnt + 1)

                if not first_peak_found:
                    # First peak is always accepted
                    peaks.append(peak_idx)
                    valleys.append(valley_idx)
                    first_peak_found = True
                    ascend_max = int(round(0.6 * fs * 0.15))
                else:
                    # Temporal distance to previous peak (sec)
                    dt_prev = (peak_idx - peaks[-1]) / fs
                    ascend_thresh = int(round((1.75 * ascend_max) / 0.6))

                    if dt_prev > 1.2 * ppi_value or ascend_cnt > ascend_thresh:
                        # Clearly a new beat
                        peaks.append(peak_idx)
                        valleys.append(valley_idx)
                        ppi_value = dt_prev
                        refractory = 0.35
                        ascend_max = int(round(0.6 * fs * 0.15))
                    else:
                        # Potentially too close – allow if beyond refractory
                        if dt_prev > refractory:
                            peaks.append(peak_idx)
                            valleys.append(valley_idx)
                            ppi_value = dt_prev
                            refractory = 0.75 * ppi_value
                            ascend_max = int(round(0.6 * ascend_cnt))
            # Reset ascent counter when slope is non‑positive
            ascend_cnt = 0

        return np.asarray(peaks, dtype=int)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _tidy_beats(self, peaks: np.ndarray, fs: float) -> np.ndarray:
        """Clean peak list by removing duplicates that are too close."""
        if peaks.size == 0:
            return peaks
        # Sort & unique for safety
        peaks = np.unique(peaks)
        min_dist = int(round(self.tidy_min_distance * fs))
        if min_dist <= 1:
            return peaks

        keep = [0]
        for idx in range(1, peaks.size):
            if peaks[idx] - peaks[keep[-1]] >= min_dist:
                keep.append(idx)
        return peaks[keep]

    def _pulse_onsets_from_peaks(
        self, sig: np.ndarray, peaks: np.ndarray, window: int | None = None
    ) -> np.ndarray:
        """Locate pulse onsets (preceding valleys) for each peak.

        Parameters
        ----------
        sig : ndarray
            PPG signal.
        peaks : ndarray[int]
            Peak indices.
        window : int, optional
            Max number of samples *before* a peak to search for the valley.  If
            *None* (default), the search extends to the midpoint between the
            current and previous peak (or the beginning of the signal for the
            first beat).
        """
        if peaks.size == 0:
            return peaks.copy()

        onsets: List[int] = []
        N = sig.size
        for k, pk in enumerate(peaks):
            if k == 0:
                start = 0
            else:
                # Search halfway between previous and current peak by default
                start = (peaks[k - 1] + pk) // 2
            if window is not None:
                start = max(start, pk - window)
            seg = sig[start : pk + 1]
            if seg.size == 0:
                onsets.append(start)
                continue
            # Valley = global minimum in segment
            rel_idx = np.argmin(seg)
            onsets.append(start + int(rel_idx))

        return np.asarray(onsets, dtype=int)
    
    def _refine_peaks(self, sig: np.ndarray, peaks: np.ndarray, fs: float,
                      window_sec: float = 0.05) -> np.ndarray:
        """
        For each candidate in `peaks`, search +/- window_sec seconds around it
        and snap to the true local maximum in that window.
        """
        w = int(window_sec * fs)
        refined = []
        N = len(sig)
        for p in peaks:
            start = max(p - w, 0)
            end   = min(p + w + 1, N)
            local = sig[start:end]
            # argmax within the local window
            offset = int(np.argmax(local))
            refined.append(start + offset)
        return np.array(refined, dtype=int)


# -----------------------------------------------------------------------------
# Convenience functional wrapper (mirrors MATLAB signature)
# -----------------------------------------------------------------------------

def mmpdv2_beat_detector(sig: Sequence[float] | np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Functional façade around :class:`MMPDV2BeatDetector` for drop‑in use."""
    detector = MMPDV2BeatDetector()
    return detector.detect(sig, fs)
