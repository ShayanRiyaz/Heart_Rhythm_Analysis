from src.lib.utils.mmpvd2 import MSPTDFastV2BeatDetector
import numpy as np
def peak_detector(seg, fs=20.83):
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