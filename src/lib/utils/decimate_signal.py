from scipy.signal import firwin, filtfilt, lfilter, resample, resample_poly
import numpy as np
from fractions import Fraction

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
    if cutoff is None or cutoff == 0.0:
        cutoff = (fs_out / 2)
    cutoff = float(cutoff)
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
