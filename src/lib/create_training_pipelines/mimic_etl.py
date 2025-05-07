import os, logging, h5py
import scipy.io as sio
import numpy as np
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


import uuid
from src.lib.utils import clean_signal, decimate_signal,find_sliding_window,scale_signal, peak_vector



class MimicETL:
    def __init__(self, config):
        # 1) Resolve and validate the input file
        self.input_dir = Path(config["input_dir"]).resolve()
        if not self.input_dir.is_file():
            raise FileNotFoundError(f"MAT file not found: {self.input_dir}")

        # 2) Derive “raw” folder automatically
        allowed_root = None
        for parent in self.input_dir.parents:
            if parent.name == "raw":
                allowed_root = parent
                break
        if allowed_root is None:
            raise ValueError(
                f"Could not find a parent folder named 'raw' for {self.input_dir}"
            )
        logger.info(f"Derived allowed raw‐data root at {allowed_root}")

        self.out_filename = str(config.get("out_filename", "etl_output"))
        # 3) Set output_dir next to it (e.g. .../data/raw → .../data/processed)
        self.output_dir = allowed_root.parent / f"processed/length_full/{self.out_filename}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.output_dir, 0o750)
        print(self.output_dir)
        # --- Pipeline parameters ---
        self.window_size_sec = int(config.get("window_size_sec", 30))
        self.fs_in = float(config.get("fs_in", 125.0))
        self.fs_out = float(config.get("fs_out", self.fs_in))
        self.bdecimate_signal = bool(config.get("decimate_signal", self.fs_in != self.fs_out))

        if self.fs_in == self.fs_out and config.get("decimate_signal", False):
            logger.warning(f"fs_in ({self.fs_in}) == fs_out ({self.fs_out}); disabling decimation.")

        self.lowpass_cutoff = float(config.get("lowpass_cutoff", 8.0))
        self.fir_numtaps = int(config.get("fir_numtaps", 33))
        self.zero_phase = bool(config.get("zero_phase", True))
        self.scale_type = config.get("scale_type")
        self.fs_ekg = float(config.get("fs_ekg", 125.0))
        self.fs_bp = float(config.get("fs_bp", 62.5))
        

        # Placeholder for HDF5 handle
        self.h5f: h5py.File

        logger.info(
            f"Initialized ETL: input={self.input_dir}, output={self.output_dir}, "
            f"fs_in={self.fs_in}, fs_out={self.fs_out}, decimate={self.bdecimate_signal}"
        )
        
    def extract(self):
        logger.info(f"Loading MAT file from {self.input_dir}")
        try:
            mat = sio.loadmat(self.input_dir, squeeze_me=True, struct_as_record=False)
        except Exception as e:
            logger.exception("Failed to load MAT file")
            raise
        return mat["data"]


    def _process_one_record(self, rec: Any) -> None:
        """
        Transform and write all windows from a single record object.
        Catches and logs per-record errors to avoid full pipeline failure.
        """
        rec_id = getattr(rec.fix, 'rec_id', 'unknown')
        subj_id = getattr(rec.fix, 'subj_id', 'unknown')
        try:
            ppg = rec.ppg.v
            fs_ppg = float(rec.ppg.fs)
            ekg = rec.ekg.v
            fs_ekg = self.fs_ekg
            abp = rec.bp.v
            fs_abp = self.fs_bp
            
            win_samples = int(self.window_size_sec * fs_ppg)
            ekg_samples = int(self.window_size_sec * fs_ekg)
            abp_samples = int(self.window_size_sec * fs_abp)
            n_windows = min(
                len(ppg) // win_samples,
                len(ekg) // ekg_samples,
                len(abp) // abp_samples
            )

            for i in range(n_windows):
                start_ppg, end_ppg = i*win_samples, (i+1)*win_samples
                start_ekg, end_ekg = i*ekg_samples, (i+1)*ekg_samples
                start_abp, end_abp = i*abp_samples, (i+1)*abp_samples

                raw_ppg = ppg[start_ppg:end_ppg]
                raw_ekg = ekg[start_ekg:end_ekg]
                raw_abp = abp[start_abp:end_abp]

                # if end_ppg > len(raw_ppg) or end_ekg > len(raw_ekg) or end_abp > len(raw_abp):
                #     continue  # skip incomplete final window

                proc = clean_signal(raw_ppg)
                if proc is None:
                    continue

                if self.bdecimate_signal:
                    proc = decimate_signal(
                        proc, fs_in=fs_ppg, fs_out=self.fs_out,
                        cutoff=self.lowpass_cutoff, numtaps=self.fir_numtaps,
                        zero_phase=self.zero_phase
                    )
                else:
                    self.fs_out = fs_ppg
                if self.scale_type is not None:
                    scaling_config = None
                    if len(proc) >= 150:
                        scaling_config = find_sliding_window(len(proc), target_windows = 5, overlap=25)
                    proc = scale_signal(proc, config = scaling_config, method = self.scale_type)
                
                peaks = np.asarray(peak_vector(proc, fs=self.fs_out))
                mask = peaks > 0 
                ref_indices = np.flatnonzero(mask)
                ref_indices = (peaks > 0).nonzero(as_tuple=True)[0]
                if len(ref_indices) < 2:
                    continue

                raw_notes = getattr(rec.fix, "subject_notes", None)
                if raw_notes is None:
                    notes = ""
                elif isinstance(raw_notes, np.ndarray):
                    if raw_notes.size == 0:
                        notes = ""
                    else:
                        arr = raw_notes.tolist()
                        notes = " ".join(str(x) for x in arr) if isinstance(arr, list) else str(arr)
                else:
                    notes = str(raw_notes)
                win = {
                    'subject': str(subj_id),
                    'rec_id': str(rec_id),
                    'window_id': str(uuid.uuid4()),
                    'raw_ppg': raw_ppg,
                    'proc_ppg': proc,
                    'raw_ekg': raw_ekg,
                    'raw_abp': raw_abp,
                    'y': peaks,
                    'ppg_fs': self.fs_out,
                    'raw_ppg_fs': fs_ppg,
                    'ekg_fs': fs_ekg,
                    'abp_fs': fs_abp,
                    'label': int(getattr(rec.fix, 'af_status', 0)),
                    'notes': notes
                }

                try:
                    self._write_window(win)
                except Exception:
                    logger.warning(f"Skipping window {win['window_id']} of record {rec_id}")
        except Exception:
            logger.exception(f"Error processing record {rec_id} for subject {subj_id}")


    def _write_window(self, win: Dict[str, Any]) -> None:
        """
        Write a single window dictionary to an open HDF5 file (self.h5f).
        Assumes self.h5f is an h5py.File opened in 'a' mode.
        """
        try:
            subj_grp = self.h5f.require_group(win['subject'])
            win_grp = subj_grp.create_group(win['window_id'])

            # datasets
            win_grp.create_dataset('raw_ppg', data=win['raw_ppg'], compression='gzip')
            win_grp.create_dataset('proc_ppg', data=win['proc_ppg'], compression='gzip')
            win_grp.create_dataset('raw_ekg', data=win['raw_ekg'], compression='gzip')
            win_grp.create_dataset('raw_abp', data=win['raw_abp'], compression='gzip')
            win_grp.create_dataset('y', data=win['y'], compression='gzip')

            # attributes
            attrs = ['ppg_fs', 'raw_ppg_fs', 'ekg_fs', 'abp_fs', 'rec_id', 'label', 'notes']
            for attr in attrs:
                win_grp.attrs[attr] = win.get(attr, '')
        except Exception:
            logger.exception(f"Failed to write window {win.get('window_id')} for subject {win.get('subject')}")
            raise

    def transform(self, recs: Any) -> None:
        """
        Process a batch of record objects by opening the HDF5 output
        and streaming each record through _process_one_record.
        """
        # ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        h5_path = self.output_dir / f"{self.out_filename}.h5"
        # open file (create if needed) in append mode
        self.h5f = h5py.File(h5_path, "a")
        try:
            for rec in np.atleast_1d(recs):
                self._process_one_record(rec)
        finally:
            self.h5f.close()

    def save_h5(self) -> Path:
        """
        Open a brand-new HDF5 (overwriting any prior run), 
        then stream every record → windows → disk via our
        existing _process_one_record + _write_window helpers.
        """
        # build the path
        h5_path = Path(self.output_dir) / f"{self.out_filename}.h5"
        logger.info(f"Creating HDF5 @ {h5_path}")

        # 'w' mode truncates any old file
        with h5py.File(h5_path, "w") as hf:
            # let our other methods see it as self.h5f
            self.h5f = hf

            # pull in all the records
            recs = self.extract()

            # ensure iterable even if single record
            for rec in np.atleast_1d(recs):
                self._process_one_record(rec)

        logger.info(f"Finished writing HDF5 with processed windows")
        return h5_path



    def process(self):
        raws = self.extract()
        self.transform(raws)
        return self.save_h5()


def main():
    mimic_num = "4"

    if mimic_num == "3":
        fs_in = 125
    elif mimic_num == "4":
        fs_in = 62.5

    # root_path = os.path.join(f'data/raw/mimic{mimic_num}_data/mimic{mimic_num}_struct.mat')
    # out_filename = f'test_mimic{mimic_num}_db'
    # out_path = os.path.join(f'../data/processed/length_full/{out_filename}')
    ver_num = 1
    root_path = os.path.join(f'data/raw/mimic{mimic_num}_data/mimic{mimic_num}_struct_v{ver_num}.mat')
    out_filename = f'mimic{mimic_num}_db_v{ver_num}'
    out_path = os.path.join(f'data/processed/length_full/{out_filename}')
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    config = {
        "input_dir": root_path,
        "output_dir":  out_path,
        "fs_in": fs_in,
        "fs_out": 20.83,
        'fs_ekg': 125,
        "window_size_sec": 8,
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