import os
import glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample
import uuid
import zarr
import h5py
import pandas as pd

class CapnoBaseETL:
    """
    ETL pipeline for CapnoBase dataset.

    Config parameters:
      input_dir: directory containing subfolder 'mat' with .mat files
      output_dir: directory to save processed segments
      target_fs: desired sampling frequency (Hz)
      window_size_sec: length of each segment (seconds)
      normalization: 'zscore', 'minmax', or 'none'
      annotation_keys: list of MAT-file keys to extract as annotations
    """
    def __init__(self, config):
        self.input_dir = config.get("input_dir", "data/mat")
        self.output_dir = config.get("output_dir", "processed_data")
        self.target_fs = config.get("target_fs", 100)
        self.window_size_sec = config.get("window_size_sec", 30)
        self.normalization = config.get("normalization", "zscore")
        self.annotation_keys = config.get("annotation_keys", ["PPG_peaks", "Resp_peaks"])

        os.makedirs(self.output_dir, exist_ok=True)
        # DataFrame to index metadata
        self.metadata_index = []


    def extract(self, filepath):
        """Load raw signals and annotations from a .mat file."""
        try:
            mat = loadmat(filepath)
            is_hdf5 = False
            
        except NotImplementedError:
            mat = h5py.File(filepath, 'r')
            is_hdf5 = True

        _inspect_mat_hdf5(filepath)
        print(is_hdf5)
        print(mat)

        data_group = f['DATA']
            
        raw = {}
        # Core signal
        raw['ppg'] = np.squeeze(data_group['PlethBySample'][()])
        print(raw)
        raw['fs'] = float(mat.get('Fs', [[self.target_fs]])[0][0])
        print(raw)
        raw['recording_id'] = os.path.splitext(os.path.basename(filepath))[0]
        print(raw)
        # Optional signals
        raw['co2'] = np.squeeze(mat.get('CO2')) if 'CO2' in mat else None
        raw['ecg'] = np.squeeze(mat.get('ECG')) if 'ECG' in mat else None
        # Annotations
        raw['annotations'] = {}

        print(raw)
        for key in self.annotation_keys:
            if key in mat:
                raw['annotations'][key] = np.squeeze(mat.get(key))
        return raw

    def transform(self, raw):
        """Resample, normalize, and segment the PPG signal."""
        ppg = raw['ppg']
        orig_fs = raw['fs']
        # Resample
        if orig_fs != self.target_fs and ppg.size > 0:
            num_samples = int(len(ppg) * self.target_fs / orig_fs)
            ppg = resample(ppg, num_samples)

        # Normalize
        if self.normalization == 'zscore':
            ppg = (ppg - np.mean(ppg)) / np.std(ppg)
        elif self.normalization == 'minmax':
            ppg = (ppg - np.min(ppg)) / (np.max(ppg) - np.min(ppg))

        # Segment into fixed windows
        samples_per_win = int(self.window_size_sec * self.target_fs)
        # print(samples_per_win)
        num_segments = len(ppg) // samples_per_win
        # print(num_segments)
        segments = []
        for i in range(num_segments):
            start = i * samples_per_win
            end = start + samples_per_win
            seg_ppg = ppg[start:end]
            print(seg_ppg)
            seg_id = str(uuid.uuid4())
            segment = {
                'ppg': seg_ppg,
                'fs': self.target_fs,
                'segment_id': seg_id,
                'recording_id': raw['recording_id'],
                'label': 'good',  # default; override via manual or heuristic labeling
                'annotations': self._slice_annotations(raw['annotations'], start, end),
            }
            segments.append(segment)
        return segments

    def _slice_annotations(self, annotations, start, end):
        """Slice annotation arrays to the current segment window."""
        sliced = {}
        for key, arr in annotations.items():
            # retain annotation indices relative to segment
            mask = (arr >= start) & (arr < end)
            if np.any(mask):
                sliced[key] = (arr[mask] - start)
            else:
                sliced[key] = np.array([])
        return sliced

    def load(self, segments):
        """Persist segments as .zarr and index metadata."""
        for seg in segments:
            fname = f"{seg['recording_id']}_{seg['segment_id']}.zarr"
            path = os.path.join(self.output_dir, fname)
            # Save PPG waveform
            zarr.save(path, seg['ppg'])
            # Record metadata
            self.metadata_index.append({
                'recording_id': seg['recording_id'],
                'segment_id': seg['segment_id'],
                'label': seg['label'],
                'annotations': seg['annotations'],
                'filepath': path
            })

    # def save_index(self):
    #     """Save metadata index as a parquet file for easy querying."""
    #     df = pd.DataFrame(self.metadata_index)
    #     index_path = os.path.join(self.output_dir, 'metadata.parquet')
    #     df.to_parquet(index_path, index=False)
    #     return df
    def save_index(self):
        """Save metadata index as an HDF5 file for easy querying."""
        df = pd.DataFrame(self.metadata_index)
        h5_path = os.path.join(self.output_dir, 'capno_base_data.h5')

        # 'metadata' is the HDF5 key / table name
        df.to_hdf(h5_path, key='capno_base_data', mode='w', format='table')
        print(df)
        # return df

    def process_all(self):
        """Run ETL for all .mat files in the input directory."""
        pattern = os.path.join(self.input_dir, '*.mat')
        # print(pattern)
        files = glob.glob(pattern)
        # print(files)
        for fp in files:
            raw = self.extract(fp)
            segments = self.transform(raw)
            self.load(segments)
        self.save_index()

    def _inspect_mat_hdf5(filepath):
        with h5py.File(filepath, 'r') as f:
            print("Top-level groups/datasets:", list(f.keys()))
            # drill down one level if you see groups
            for name in f.keys():
                try:
                    sub = f[name]
                    if isinstance(sub, h5py.Group):
                        print(f"Contents of group '{name}':", list(sub.keys()))
                except Exception:
                    pass
        # return df
# if __name__ == "__main__":
#     # Default paths; adjust as needed or wrap with argparse
#     root_path = os.path.join('data/raw/capnobase/data/mat')
#     out_path = os.path.join('data/processed/length_full')
#     # root_path = os.getenv('CAPNOBASE_ROOT', 'data/mat')
#     # out_path = os.getenv('CAPNOBASE_OUT', 'processed_data')
#     config = {
#         'input_dir': root_path,
#         'output_dir': out_path,
#         'target_fs': 125,
#         'window_size_sec': 30,
#         'normalization': 'zscore',
#         'annotation_keys': ['PPG_peaks', 'Resp_peaks']
#     }
#     etl = CapnoBaseETL(config)
#     etl.process_all()
