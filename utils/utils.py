import torch
from peak_detectors.mmpvd2 import MSPTDFastV2BeatDetector
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import h5py


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


def scale_signal(input_vector: torch.Tensor,
                 config: tuple,
                 window_fn=None,
                 method: str = 'norm') -> torch.Tensor:
    """
    Applies standardization or normalization over sliding windows defined by config,
    and reconstructs the full signal via overlap-add.

    Args:
        input_vector (torch.Tensor): 1D tensor of shape (N,).
        config (tuple): (window_size, hop_size, num_windows).
        window_fn (callable, optional): fn(length) -> torch.Tensor of weights.
                                         Defaults to uniform weights.
        method (str): 'standard' for z-score, 'norm' for min-max then zero-mean.

    Returns:
        torch.Tensor: standardized tensor of same shape.
    """
    device = input_vector.device
    dtype = input_vector.dtype
    input_size = input_vector.size(0)
    window_size, hop_size, _ = config

    if window_fn is None:
        window_fn = lambda n: torch.ones(n, device=device, dtype=dtype)

    out = torch.zeros(input_size, device=device, dtype=dtype)
    weights = torch.zeros(input_size, device=device, dtype=dtype)

    for start in range(0, input_size, hop_size):
        end = min(start + window_size, input_size)
        segment = input_vector[start:end]

        if method == 'standard':
            mu = segment.mean()
            sigma = segment.std(unbiased=False)
            if sigma == 0:
                z = torch.zeros_like(segment)
            else:
                z = (segment - mu) / sigma

        elif method == 'norm':
            min_val = segment.min()
            max_val = segment.max()
            denom = max_val - min_val
            if denom == 0:
                z = torch.zeros_like(segment)
            else:
                z = (segment - min_val) / denom
                z = z - z.mean()
        else:
            raise ValueError(f"Unknown method '{method}'")

        w = window_fn(end - start)
        if not torch.is_tensor(w):
            w = torch.tensor(w, device=device, dtype=dtype)

        out[start:end] += z * w
        weights[start:end] += w

        if end == input_size:
            break

    mask = weights > 0
    out[mask] = out[mask] / weights[mask]
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
    # y = ndi.gaussian_filter1d(y, sigma=1, mode='constant')
    return y


class PPGWindow(Dataset):
    def __init__(self, h5_path, win_len=625, transform=True,var_mult=500,best_config=(100, 75, 8),scale_type="standard"):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r', swmr=True)
        # self.ids = list(self.h5.keys())
        self.ids = [gid for gid in self.h5.keys() if 'ppg' in self.h5[gid]]
        self.win = win_len
        self.transform = transform
        self.var_mult = var_mult
        self.best_config = best_config
        self.scale_type = scale_type

        # Precompute valid window-start indices for each record
        self.valid_starts = {}
        for gid in self.ids:
            arr = self.h5[gid]['ppg'][:].astype('float32')
            # pad wrap if too short
            if len(arr) < self.win + 1:
                arr = np.pad(arr, (0, self.win + 1 - len(arr)), mode='wrap')

            ends = len(arr) - self.win + 1

            # vectorized check: slide a window of length win
            # create strided view for mean and std
            # fallback simple loop if memory is a concern
            
            starts = []
            for s in range(0, ends):
                seg = arr[s:s+self.win]
                if seg.any() and seg.std() != 0:
                    starts.append(s)
            self.valid_starts[gid] = np.array(starts, dtype=np.int64)
            
    def __len__(self):
        # virtually infinite; adjust if you want a fixed number
        return len(self.ids) * self.var_mult

    def __getitem__(self, idx):
        # select record and then a valid start index
        gid = self.ids[idx % len(self.ids)]
        starts = self.valid_starts[gid]
        label = self.h5[gid].attrs.get('af_status')
        if len(starts) == 0:
            # no valid windows: fallback to uniform sampling
            x = self.h5[gid]['ppg'][:].astype('float32')
            if len(x) < self.win + 1:
                x = np.pad(x, (0, self.win+1-len(x)), mode='wrap')
            s = 0
        else:
            s = int(np.random.choice(starts))
            x = self.h5[gid]['ppg'][:].astype('float32')
            if len(x) < self.win + 1:
                x = np.pad(x, (0, self.win+1-len(x)), mode='wrap')

        seg_np = x[s:s+self.win]
        if sum(np.diff(seg_np)) == 0:
        # throw away this sample, pick the “next” one instead
            return self.__getitem__((idx+1) % len(self))
        
        seg = torch.from_numpy(seg_np)
        seg = scale_signal(seg, self.best_config, method= self.scale_type).float()


        y_np = pseudo_peak_vector(seg.cpu().numpy())    # convert back to numpy
        y = torch.from_numpy(y_np).float()
        return (seg.unsqueeze(0), y, torch.from_numpy(seg_np).unsqueeze(0).float(),label)

    def __getstate__(self):
        # drop the actual file handle before pickling
        state = self.__dict__.copy()
        state.pop('h5', None)
        return state

    def __setstate__(self, state):
        # restore everything, then re‑open HDF5
        self.__dict__.update(state)
        self.h5 = h5py.File(self.h5_path, 'r', swmr=True)