import numpy as np

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
