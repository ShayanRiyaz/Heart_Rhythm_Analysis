import numpy as np
from torch.utils.data import Dataset
import torch
import h5py

class PPGWindowLoader(Dataset):
    def __init__(self, h5_path, win_len: int = 625, transform=None):
        """
        h5_path: path to your windowed HDF5
        win_len: the length you want every window to be
        transform: optional callable applied to proc_ppg Tensor
        """
        self.h5_path  = h5_path
        self.win_len  = win_len
        self.transform = transform

        # open HDF5
        self.h5 = h5py.File(h5_path, 'r', swmr=True)

        # build flat index of (subject, window_id)
        self.index = [
            (subj, wid)
            for subj in self.h5.keys()
            for wid  in self.h5[subj].keys()
        ]

    def __len__(self):
        return len(self.index)

    def _pad_or_trim(self, arr: np.ndarray) -> np.ndarray:
        L = arr.shape[0]
        if L < self.win_len:
            # wrap-pad
            return np.pad(arr, (0, self.win_len - L), mode='wrap')
        else:
            return arr[:self.win_len]

    def __getitem__(self, idx):
        subj, wid = self.index[idx]
        grp       = self.h5[subj][wid]

        # load processed PPG and raw PPG
        proc = grp['proc_ppg'][:]    # float32, may vary length
        raw  = grp['raw_ppg'][:]     # float64, full trace
        # print(raw.shape)
        # load y as binary mask
        y_mask = grp['y'][:]         # float32 mask of peaks

        # pad/trim to win_len
        proc   = self._pad_or_trim(proc).astype('float32')
        raw    = self._pad_or_trim(raw).astype('float32')
        mask   = self._pad_or_trim(y_mask).astype('float32')

        # convert to tensors
        proc_t = torch.from_numpy(proc).float().unsqueeze(0)  # (1, win_len)
        raw_t  = torch.from_numpy(raw).float().unsqueeze(0)   # (1, win_len)
        mask_t = torch.from_numpy(mask).float()               # (win_len,)

        # optional transform on proc
        if self.transform:
            proc_t = self.transform(proc_t)

        # peak count (number of ones in mask)
        peak_count = torch.tensor(int(mask_t.sum().item()), dtype=torch.long)

        # metadata
        label  = grp.attrs.get('label')
        raw_fs = grp.attrs.get('raw_ppg_fs')

        return proc_t, mask_t, peak_count#,raw_t , label, raw_fs

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('h5', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.h5 = h5py.File(self.h5_path, 'r', swmr=True)

# class PPGWindowLoader(Dataset):
#     def __init__(self, h5_path, win_len: int = 625, transform=None, max_peak_count=20):
#         """
#         h5_path: path to your windowed HDF5
#         win_len: the length you want every window to be
#         transform: optional callable applied to proc_ppg Tensor
#         max_peak_count: maximum number of peaks to pad to
#         """
        
#         self.h5_path = h5_path
#         self.win_len = win_len
#         self.h5 = h5py.File(h5_path, 'r', swmr=True)
#         self.transform = transform
#         self.max_peak_count = max_peak_count

#         # build flat index
#         print("Building dataset index...")
#         self.index = [
#             (subj, wid)
#             for subj in self.h5.keys()
#             for wid in self.h5[subj].keys()
#         ]
#         print(f"Dataset contains {len(self.index)} samples")

#     def __len__(self):
#         return len(self.index)

#     def _pad_or_trim(self, arr: np.ndarray):
#         L = arr.shape[0]
#         if L < self.win_len:
#             # wrap-pad if you like, or constant-pad with zero
#             return np.pad(arr, (0, self.win_len - L), mode='wrap')
#         else:
#             return arr[:self.win_len]

#     def __getitem__(self, idx):
#         subj, wid = self.index[idx]
#         grp = self.h5[subj][wid]

#         # raw arrays from file
#         proc = grp['proc_ppg'][:]    # float32, length maybe ≠ win_len
#         y = grp['y'][:]              # float32, same (this contains the peaks)
        
#         # enforce uniform length
#         proc = self._pad_or_trim(proc).astype('float32')
#         y = self._pad_or_trim(y).astype('float32')

#         # to torch
#         proc_t = torch.from_numpy(proc).float().unsqueeze(0)  # (1, win_len)
#         y_t = torch.from_numpy(y).float()                    # (win_len,)

#         # Extract peak positions from y signal
#         # Assuming peaks are marked with 1.0 in the y signal
#         peak_indices = np.where(y > 0.5)[0]
        
#         # Pad peak indices to fixed length
#         peak_count = len(peak_indices)
#         padded_peaks = np.full(self.max_peak_count, -1, dtype=np.int64)
#         padded_peaks[:min(peak_count, self.max_peak_count)] = peak_indices[:min(peak_count, self.max_peak_count)]
        
#         # Convert to torch tensors
#         peak_positions = torch.tensor(padded_peaks, dtype=torch.long)
#         peak_count = torch.tensor(min(peak_count, self.max_peak_count), dtype=torch.long)

#         if self.transform:
#             proc_t = self.transform(proc_t)
#         print(proc_t.shape, peak_positions.shape)
#         return proc_t, peak_positions, peak_count

#     def __getstate__(self):
#         st = self.__dict__.copy()
#         st.pop('h5', None)
#         return st

#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         self.h5 = h5py.File(self.h5_path, 'r', swmr=True)