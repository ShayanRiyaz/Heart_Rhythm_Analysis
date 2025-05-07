from typing import Callable, Tuple, Optional
import numpy as np

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