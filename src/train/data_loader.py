import os
import torch
from torch.utils.data import DataLoader,random_split
from lib.create_training_pipelines.ppg_window_loader import PPGWindowLoader
from train.config import config

def custom_collate(batch):
    x_list, y_list, cnt_list = zip(*batch)
    x_batch = torch.stack(x_list, dim=0)    # (B,1,167)
    y_batch    = torch.stack(y_list,  dim=0)      # (B,167)
    peak_count_batch  = torch.stack(cnt_list, dim=0)     # (B,20)
    # cnt_batch  = torch.stack(cnt_list, dim=0)     # (B,)
    return x_batch, y_batch, peak_count_batch

def get_datasets(config,win_len=None):
    win_len = win_len or config.WIN_LEN
    train_ds = PPGWindowLoader(config.paths.TRAIN_LOADER_PATH, win_len=win_len)
    # val_ds   = PPGWindowLoader(VAL_LOADER_PATH,   win_len=win_len)
    VAL_FRAC = 0.20               # 10 % of the *virtual* samples
    n_total  = len(train_ds)
    n_val    = int(n_total * VAL_FRAC)
    n_train  = n_total - n_val

    train_ds, val_ds = random_split(
        train_ds,
        lengths=[n_train, n_val],
        generator=torch.Generator().manual_seed(42)   # reproducible
    )

    test_ds  = PPGWindowLoader(config.paths.TEST_LOADER_PATH,  win_len=win_len)
    return train_ds, val_ds, test_ds


def get_dataloaders(config):
    train_ds, val_ds, test_ds = get_datasets(config)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH,
        shuffle=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH,
        shuffle=False,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH,
        shuffle=False,
        collate_fn=custom_collate
    )
    return train_loader, val_loader, test_loader