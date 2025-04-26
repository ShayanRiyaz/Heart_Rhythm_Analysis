import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import uuid
import zarr
import h5py
import pandas as pd
from utils.utils import decimate_signal,clean_signal,find_sliding_window,scale_signal,pseudo_peak_vector

class CapnoBaseETL:
    """
    ETL pipeline for CapnoBase: window → clean → decimate → store in single HDF5
    """
    def __init__(self, config):
        self.input_dir = config.get("input_dir", "data/mat")
        self.output_dir = config.get("output_dir", "processed_data")
        self.window_size_sec = config.get("window_size_sec", 30)
        self.fs_in = config.get("fs_in", 125.0)
        self.fs_out = config.get("fs_out", 20.83)
        self.lowpass_cutoff = config.get("lowpass_cutoff", 8.0)
        self.fir_numtaps = config.get("fir_numtaps", 129)
        self.zero_phase = config.get("zero_phase", True)
        self.out_filename = config.get("out_filename", 'CapnoBaseETL')
        self.windows_data = []  # list of dicts
        self.scale_type = "norm"

    def extract(self, filepath):
        try:
            mat = loadmat(filepath)
            is_hdf5 = False
        except NotImplementedError:
            mat = h5py.File(filepath, 'r')
            is_hdf5 = True
        if is_hdf5:
            f = mat
            grp = f['signal']
            pleth_obj = grp['pleth']
            if isinstance(pleth_obj, h5py.Dataset):
                ppg = pleth_obj[()].squeeze()
            else:
                ds_name = next((k for k,v in pleth_obj.items() if isinstance(v, h5py.Dataset)), None)
                ppg = pleth_obj[ds_name][()].squeeze()
            sr_obj = f['param']['samplingrate']
            if isinstance(sr_obj, h5py.Dataset):
                sr_data = sr_obj[()]
            else:
                ds = next(v for v in sr_obj.values() if isinstance(v, h5py.Dataset))
                sr_data = ds[()]
            fs = float(np.array(sr_data).flat[0])
            f.close()
        else:
            ppg = np.squeeze(mat.get('Pleth', np.array([])))
            fs = float(mat.get('Fs', [[self.fs_in]])[0][0])
        return {'ppg': ppg, 'fs': fs,
                'subject': os.path.splitext(os.path.basename(filepath))[0]}

    def transform(self, raw):
        ppg, fs = raw['ppg'], raw['fs']
        win_samples = int(self.window_size_sec * fs)
        for i in range(len(ppg) // win_samples):
            start, end = i*win_samples, (i+1)*win_samples
            raw_win = ppg[start:end]
            cleaned = clean_signal(raw_win)
            if cleaned is None: continue
            dec = decimate_signal(cleaned,
                                 fs_in=fs, fs_out=self.fs_out,
                                 cutoff=self.lowpass_cutoff,
                                 numtaps=self.fir_numtaps,
                                 zero_phase=self.zero_phase)
            scaling_config = find_sliding_window(len(dec), target_windows = 5, overlap=25)
            x = scale_signal(dec, config = scaling_config, method = self.scale_type)
            y_peaks = pseudo_peak_vector(dec)    # convert back to numpy
            
            win_id = str(uuid.uuid4())
            self.windows_data.append({
                'subject': raw['subject'],
                'rec_id' : 0,
                'window_id': win_id,
                'raw_ppg': raw_win,
                'proc_ppg': x,
                'y':y_peaks,
                'fs': self.fs_out,
                'label':-1
            })


    def save_h5(self):
        h5_path = os.path.join(self.output_dir, f'{self.out_filename}.h5')
        with h5py.File(h5_path, 'w') as hf:
            for win in self.windows_data:
                subj_grp = hf.require_group(win['subject'])
                win_grp = subj_grp.create_group(win['window_id'])
                win_grp.create_dataset('raw_ppg', data=win['raw_ppg'], compression='gzip')
                win_grp.create_dataset('proc_ppg', data=win['proc_ppg'], compression='gzip')
                win_grp.create_dataset('y', data=win['y'], compression='gzip')
                win_grp.attrs['fs'] = win['fs']
                win_grp.attrs['rec_id'] = win['rec_id']
                win_grp.attrs['label'] = win['label']
        return h5_path

    def process_all(self):
        for fp in glob.glob(os.path.join(self.input_dir, '*.mat')):
            raw = self.extract(fp)
            self.transform(raw)
        return self.save_h5()