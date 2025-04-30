
import h5py
import os
import numpy as np
from typing import List

def load_all_windows_from_h5(file_paths: List[str]):
    all_windows = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as hf:
            for subject_id in hf.keys():
                subj_group = hf[subject_id]
                for window_id in subj_group.keys():
                    win_group = subj_group[window_id]
                    window_data = {
                        "subject_id": subject_id,
                        "window_id": window_id,
                        "data": {k: win_group[k][()] for k in win_group.keys()},
                        "attrs": {k: win_group.attrs[k] for k in win_group.attrs}
                    }
                    all_windows.append(window_data)
    return all_windows

def save_windows_to_h5(windows, output_path):
    with h5py.File(output_path, 'w') as hf:
        for win in windows:
            group = hf.require_group(f"{win['subject_id']}/{win['window_id']}")
            for key, value in win['data'].items():
                group.create_dataset(key, data=value)
            for attr_key, attr_val in win['attrs'].items():
                group.attrs[attr_key] = attr_val

def split_and_save(file_paths: List[str], train_ratio: float = 0.8, output_dir: str = "."):
    all_windows = load_all_windows_from_h5(file_paths)
    split_index = int(len(all_windows) * train_ratio)
    train_windows = all_windows[:split_index]
    test_windows = all_windows[split_index:]

    train_path = os.path.join(output_dir, "train_dataset.h5")
    test_path = os.path.join(output_dir, "test_dataset.h5")

    save_windows_to_h5(train_windows, train_path)
    save_windows_to_h5(test_windows, test_path)

    return train_path, test_path
