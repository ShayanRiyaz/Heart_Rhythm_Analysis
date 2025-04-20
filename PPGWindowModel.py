from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import torch
from utils import *

class PPGWindow(Dataset):
    def __init__(self, h5_path, win_len=625, transform=True,var_mult=500,best_config=(100, 75, 8),scale_type="standard"):
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

        seg = torch.from_numpy(seg_np)
        seg = scale_signal(seg, self.best_config, method= self.scale_type).float()


        y_np = pseudo_peak_vector(seg.cpu().numpy())    # convert back to numpy
        y = torch.from_numpy(y_np).float()
        return (seg.unsqueeze(0), y, torch.from_numpy(seg_np).unsqueeze(0).float(),label)
