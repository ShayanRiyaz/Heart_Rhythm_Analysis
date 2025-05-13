import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import uuid
import zarr
import h5py
import pandas as pd
from src.lib.utils.mmpvd2 import MSPTDFastV2BeatDetector
from src.lib.utils import decimate_signal,clean_signal,find_sliding_window,scale_signal,peak_detector

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
        self.scale_type = config.get("scale_type", None)
        self.bdecimate_signal = None 

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
            # PPG
            pleth_obj = grp['pleth']
            if isinstance(pleth_obj, h5py.Dataset):
                ppg = pleth_obj[()].squeeze()
            else:
                ds_name = next((k for k,v in pleth_obj.items() if isinstance(v, h5py.Dataset)), None)
                ppg = pleth_obj[ds_name][()].squeeze()
            sr_obj = f['param']['samplingrate']['pleth']
            if isinstance(sr_obj, h5py.Dataset):
                sr_data = sr_obj[()]
            else:
                ds = next(v for v in sr_obj.values() if isinstance(v, h5py.Dataset))
                sr_data = ds[()]
            ppg_fs = float(np.array(sr_data).flat[0])

            # ECG
            ekg_obj = grp['ecg']
            if isinstance(ekg_obj, h5py.Dataset):
                ekg = ekg_obj[()].squeeze()
            else:
                ds_name = next((k for k,v in ekg_obj.items() if isinstance(v, h5py.Dataset)), None)
                ekg = ekg_obj[ds_name][()].squeeze()
            sr_obj = f['param']['samplingrate']['ecg']
            if isinstance(sr_obj, h5py.Dataset):
                sr_data = sr_obj[()]
            else:
                ds = next(v for v in sr_obj.values() if isinstance(v, h5py.Dataset))
                sr_data = ds[()]
            ekg_fs = float(np.array(sr_data).flat[0])

            f.close()
        else:
            ppg = np.squeeze(mat.get('Pleth', np.array([])))
            ekg = np.squeeze(mat.get('ecg', np.array([])))
            ppg_fs = float(mat.get('ppg_fs', [[self.fs_in]])[0][0])
            ppg_fs = float(mat.get('ekg_fs', [[self.fs_in]])[0][0])
        return {'ppg': ppg, 
                'ppg_fs': ppg_fs,
                'ekg': ekg,
                'ekg_fs': ekg_fs,
                'subject': os.path.splitext(os.path.basename(filepath))[0]}

    def transform(self, raw):
        ppg, ppg_fs,ekg,ekg_fs = raw['ppg'], raw['ppg_fs'],raw['ekg'], raw['ekg_fs']
        win_samples = int(self.window_size_sec * ppg_fs)
        notes = ""
        for i in range(len(ppg) // win_samples):
            start, end = i*win_samples, (i+1)*win_samples
            raw_win = ppg[start:end]
            raw_ekg = ekg[start:end]
            x = clean_signal(raw_win)
            if x is None: continue
            if (self.bdecimate_signal) is False or (self.bdecimate_signal is None):
                x = decimate_signal(x,
                                    fs_in=ppg_fs, fs_out=self.fs_out,
                                    cutoff=self.lowpass_cutoff,
                                    numtaps=self.fir_numtaps,
                                    zero_phase=self.zero_phase)
                
            else: 
                self.fs_out = ppg_fs
            if self.scale_type is not None:
                scaling_config = find_sliding_window(len(x), target_windows = 5, overlap=25)
                x = scale_signal(x, config = scaling_config, method = self.scale_type)
            y_peaks = pseudo_peak_vector(x,fs = self.fs_out)    # convert back to numpy
            
            win_id = str(uuid.uuid4())
            self.windows_data.append({
                'subject': raw['subject'],
                'rec_id' : 0,
                'window_id': win_id,
                'raw_ppg': raw_win,
                'proc_ppg': x,
                'y':y_peaks,
                'raw_ekg': raw_ekg,
                'ppg_fs': self.fs_out,
                'ekg_fs': ekg_fs,
                'label':-1,
                'notes':notes
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
                win_grp.create_dataset('raw_ekg', data=win['raw_ekg'], compression='gzip')
                win_grp.attrs['ppg_fs'] = win['ppg_fs']
                win_grp.attrs['ekg_fs'] = win['ekg_fs']
                win_grp.attrs['rec_id'] = win['rec_id']
                win_grp.attrs['label'] = win['label']
                win_grp.attrs['notes'] = win['notes']
        return h5_path

    def process_all(self):
        for fp in glob.glob(os.path.join(self.input_dir, '*.mat')):
            raw = self.extract(fp)
            self.transform(raw)
        return self.save_h5()
    
def main():
    root_path = os.path.join(f'{os.getcwd()}/data/raw/capnobase/data/mat')
    out_path = os.path.join(f'{os.getcwd()}/data/processed/length_full/capnobase_db')
    out_filename = 'capnobase_db'
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    fs_in = 100.00
    fs_out = 20.83

    config = {
    "input_dir"      : root_path,
    "output_dir"     : out_path,
    "window_size_sec": 30,
    "fs_in"          : fs_in,
    "fs_out"   : fs_out,
    "lowpass_cutoff" : (fs_out / 2),
    "fir_numtaps"    : 129,
    "scale_type": None,
    "decimate_signal": None,
    "zero_phase"     : True,
    "out_filename" :  out_filename
}
    # cfg = {'input_dir': root_path, 'output_dir': out_path}
    etl = CapnoBaseETL(config)
    h5file = etl.process_all()
    print(f"Saved windows HDF5 to {h5file}")
    # df = load_as_df(out_path,out_filename)

    # display(df.head(10))
    return

if __name__ == "__main__":
    main()