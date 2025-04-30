import os
import scipy.io as sio
import numpy as np
import h5py

import uuid
from heart_rhythm_analysis.utils.utils import clean_signal, decimate_signal,find_sliding_window,scale_signal,pseudo_peak_vector

class MimicETL:
    def __init__(self, config):
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]
        self.window_size_sec = config.get("window_size_sec", 30)
        self.fs_in = config.get("fs_in", 125.0)
        self.fs_out = config.get("fs_out",self.fs_in)
        self.lowpass_cutoff = config.get("lowpass_cutoff", 8.0)
        self.fir_numtaps = config.get("fir_numtaps", 129)
        self.zero_phase = config.get("zero_phase", True)
        self.out_filename = config.get("out_filename", True)
        self.windows_data = []  # list of dicts
        self.scale_type = config.get("scale_type", None)
        self.bdecimate_signal =  config.get("decimate_signal", False)
        self.fs_ekg = config.get("fs_ekg",125)
        self.fs_bp = config.get("fs_bp",62.5)
        
    def extract(self):
        print(f"Loading {self.input_dir}")
        mat = sio.loadmat(self.input_dir, squeeze_me=True, struct_as_record=False)
        return mat["data"]

    def transform(self, recs):
        windows = []
        for rec in np.atleast_1d(recs):
            rec_id = str(rec.fix.rec_id)
            subj_id = str(rec.fix.subj_id)
            af_status = int(rec.fix.af_status)
            try:
                notes = rec.fix.notes
            except Exception as E:
                notes = ""
            
            sig_obj = getattr(rec, "ppg", None)
            if not hasattr(sig_obj, "v"):
                continue

            fs = float(sig_obj.fs)
            win_samples = int(self.window_size_sec * fs)
            ppg_data = sig_obj.v

            ekg_obj = getattr(rec, "ekg", None)
            if not hasattr(ekg_obj, "v"):
                continue
            ekg_data  = ekg_obj.v
            ekg_fs =self.fs_ekg #float(ekg_obj.fs)
            ekg_win_samples = int(self.window_size_sec * ekg_fs)

            abp_obj = getattr(rec, "bp", None)
            if not hasattr(abp_obj, "v"):
                continue
            abp_data  = abp_obj.v
            abp_fs =self.fs_bp;#float(abp_obj.fs)
            abp_win_samples = int(self.window_size_sec * abp_fs)
            for i in range(len(ppg_data) // win_samples):
                start, end = i*win_samples, (i+1)*win_samples
                start_ekg,end_ekg = i*ekg_win_samples, (i+1)*ekg_win_samples
                start_abp,end_abp = i*abp_win_samples, (i+1)*abp_win_samples

                raw_win = ppg_data[start:end]
                raw_ekg = ekg_data[start_ekg:end_ekg]
                raw_abp = abp_data[start_abp:end_abp]

                x = clean_signal(raw_win)
                if x is None: continue
                if (self.bdecimate_signal is True):
                    x = decimate_signal(x,
                                        fs_in=fs, fs_out=self.fs_out,
                                        cutoff=self.lowpass_cutoff,
                                        numtaps=self.fir_numtaps,
                                        zero_phase=self.zero_phase)
                else: 
                    self.fs_out = fs

                if self.scale_type is not None:
                    scaling_config = find_sliding_window(len(x), target_windows = 5, overlap=25)
                    x = scale_signal(x, config = scaling_config, method = self.scale_type)
                y_peaks = pseudo_peak_vector(x,fs = self.fs_out )    # convert back to numpy
                win_id = str(uuid.uuid4())
                self.windows_data.append({
                    'subject': f'{subj_id}',
                    'rec_id': f'{rec_id}',
                    'window_id': win_id,
                    'ppg_fs': self.fs_out,
                    'raw_ppg': raw_win,
                    'proc_ppg': x,
                    'y':y_peaks,
                    'ekg_fs': ekg_fs,
                    'raw_ekg': raw_ekg,
                    'raw_abp':raw_abp,
                    'abp_fs':abp_fs,
                    'label': af_status,
                    'notes': notes
                })

    def save_h5(self):
        h5_path = os.path.join(self.output_dir, f'{self.out_filename}.h5')
        with h5py.File(h5_path, 'w') as hf:
            for win in self.windows_data:
                subj_grp = hf.require_group(win['subject'])
                win_grp = subj_grp.create_group(win['window_id'])
                win_grp.create_dataset('raw_ppg', data=win['raw_ppg'], compression='gzip')
                win_grp.create_dataset('proc_ppg', data=win['proc_ppg'], compression='gzip')
                win_grp.create_dataset('raw_ekg', data=win['raw_ekg'], compression='gzip')
                win_grp.create_dataset('raw_abp', data=win['raw_abp'], compression='gzip')
                win_grp.create_dataset('y', data=win['y'], compression='gzip')
                win_grp.attrs['ppg_fs'] = win['ppg_fs']
                win_grp.attrs['ekg_fs'] = win['ekg_fs']
                win_grp.attrs['abp_fs'] = win['abp_fs']
                win_grp.attrs['rec_id'] = win['rec_id']
                win_grp.attrs['label'] = win['label']
                try:
                    win_grp.attrs['notes'] = win['notes']
                except:
                    win_grp.attrs['notes'] = ""
        return h5_path


    def process(self):
        raws = self.extract()
        self.transform(raws)
        return self.save_h5()


def main():
    mimic_num = "3"
    root_path = os.path.join(f'data/raw/mimic{mimic_num}_data/_mimic{mimic_num}_struct.mat')
    out_filename = f'mimic{mimic_num}_db'
    out_path = os.path.join(f'../data/processed/length_full/{out_filename}')
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    config = {
        "input_dir": root_path,
        "output_dir":  out_path,
        "fs_in": 125.00,
        "fs_out": 20.83,
        'fs_ekg': 125,
        "window_size_sec": 30,
        "scale_type": "norm",
        "decimate_signal": True,
        "zero_phase": True,
        "out_filename": out_filename 
        
    }
    bSetUpDB = True
    if bSetUpDB:
        etl = MimicETL(config)
        out_file = etl.process()
        print("Saved General MIMIC III windows to", out_file)
    return
if __name__ == "__main__":
    main()