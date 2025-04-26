import os
import scipy.io as sio
import numpy as np
import h5py
import uuid
from utils.utils import clean_signal, decimate_signal,find_sliding_window,scale_signal,pseudo_peak_vector

class MimicAFETL:
    def __init__(self, config):
        self.input_dir = config["input_dir"]
        self.output_dir = config["output_dir"]
        self.window_size_sec = config.get("window_size_sec", 30)
        self.fs_in = config.get("fs_in", 125.0)
        self.fs_out = config.get("fs_out", 20.83)
        self.lowpass_cutoff = config.get("lowpass_cutoff", 8.0)
        self.fir_numtaps = config.get("fir_numtaps", 129)
        self.zero_phase = config.get("zero_phase", True)
        self.out_filename = config.get("out_filename", True)
        self.windows_data = []  # list of dicts
        self.scale_type = "norm"
        
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

            sig_obj = getattr(rec, "ppg", None)
            if not hasattr(sig_obj, "v"):
                continue

            data = clean_signal(sig_obj.v)
            if data is None:
                continue
            ppg = data 
            fs = float(sig_obj.fs)
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
                    'subject': f'{subj_id}',
                    'rec_id': f'{rec_id}',
                    'window_id': win_id,
                    'raw_ppg': raw_win,
                    'proc_ppg': x,
                    'y':y_peaks,
                    'fs': self.fs_out,
                    'label': af_status
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


    def process(self):
        raws = self.extract()
        self.transform(raws)
        return self.save_h5()

