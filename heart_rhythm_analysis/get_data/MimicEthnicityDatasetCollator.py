import os
import requests
import gzip
import shutil
import pandas as pd
import numpy as np
import wfdb
from scipy.io import savemat
from concurrent.futures import ThreadPoolExecutor

class MimicEthnicityDatasetCollator:
    def __init__(self, config):
        self.config = config
        
        self.root_dir = config["paths"]["local"]["root_folder"]
        self.temp_dir = os.path.join(self.root_dir, "downloaded_files")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.ethnicity_extract = config.get("ethnicity_extract", False)
        self.num_subjects =config.get("num_subjects", 200)
        self.ethnicity_data = None
        self.out_path = os.path.join(self.root_dir, "mimic_ethnicity_struct_output.mat")

        self.ecg_ppg_labels = ['PLETH','II']
        self.Arterial_blood_pressure_labels = ['ABP','ART']

    def collate_dataset(self, load_metadata=True, load_waveforms=True, download_waveforms=False):
        print("\n ~~~ Downloading and collating MIMIC III matched waveform subset ~~~")

        self.setup_paths()

        if load_metadata and self.ethnicity_extract:
            self.ensure_metadata_downloaded()
            self.load_ethnicity_data()
        else:
            print("Skipping metadata download/extraction.")
            self.ethnicity_data = None

        records = []
        if download_waveforms:
            records = self.download_dataset()

        if load_waveforms:
            self.scan_waveform_directory(records,)

    def setup_paths(self):
        print("\n - Setting up parameters")
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root folder does not exist: {self.root_dir}")
        print("Working directory verified:", self.root_dir)

    def download_and_extract_gz(self, url, out_path):
        print(f"Downloading from {url}")
        gz_path = out_path + ".gz"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(gz_path, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(gz_path)

    def ensure_metadata_downloaded(self):
        admissions_path = os.path.join(self.temp_dir, "ADMISSIONS.csv")
        patients_path = os.path.join(self.temp_dir, "PATIENTS.csv")

        if not os.path.exists(admissions_path):
            url = "https://physionet.org/static/published-projects/mimiciii/1.4/ADMISSIONS.csv.gz"
            self.download_and_extract_gz(url, admissions_path)

        if not os.path.exists(patients_path):
            url = "https://physionet.org/static/published-projects/mimiciii/1.4/PATIENTS.csv.gz"
            self.download_and_extract_gz(url, patients_path)

    def load_ethnicity_data(self):
        print("Loading ethnicity data...")
        admissions_path = os.path.join(self.temp_dir, "ADMISSIONS.csv")
        patients_path = os.path.join(self.temp_dir, "PATIENTS.csv")

        admissions = pd.read_csv(admissions_path)
        patients = pd.read_csv(patients_path)

        df = pd.merge(admissions[['SUBJECT_ID', 'ETHNICITY']],
                      patients[['SUBJECT_ID', 'GENDER', 'DOB']],
                      on='SUBJECT_ID')
        self.ethnicity_data = df

    def download_dataset(self):
        print("Downloading waveform records...")
        records_file = os.path.join(self.temp_dir, "RECORDS")

        if not os.path.exists(records_file):
            url = "https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/RECORDS"
            r = requests.get(url)
            r.raise_for_status()
            with open(records_file, "w") as f:
                f.write(r.text)

        with open(records_file, "r") as f:
            records = [line.strip() for line in f.readlines()]
        return records
        # for rec in records[:10]:  # Still limiting for test
        #     try:
        #         wfdb.dl_database(
        #             "mimic3wdb-matched",
        #             dl_dir=self.temp_dir,
        #             records=[rec],
        #             keep_subdirs=True
        #         )
        #     except Exception as e:
        #         print(f"Failed to download record {rec}: {e}")

    def get_file_extensions(self,remote_url):
        from bs4 import BeautifulSoup
        import requests

        r = requests.get(remote_url)
        soup = BeautifulSoup(r.text, 'html.parser')
        return [a['href'] for a in soup.find_all('a') if any(a['href'].endswith(ext) for ext in ['.hea', '.dat', '.atr'])]
    
    def load_header(self, base_name, pn_dir):
        try:
            return wfdb.rdheader(record_name=base_name, pn_dir=pn_dir)
        except Exception as e:
            print(f"Failed to load header {base_name}: {e}")
            return None
        
    def scan_waveform_directory(self,records_base):
        print("Scanning for waveform files...")
        curr_subject_count = 0
        subject_data = []
        for curr_record in records_base:
            if curr_subject_count >= self.num_subjects:
                break

            # curr_dir = f"https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/{curr_record[:-1]}"
            # file_recs = self.get_file_extensions(curr_dir)
            # hea_files = [f for f in file_recs if f.endswith('.hea')]

            curr_dir = curr_record.rstrip('/')
            full_url = f"https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/{curr_dir}"
            file_recs = self.get_file_extensions(full_url)
            hea_files = [f for f in file_recs if f.endswith('.hea')]

            curr_pn_dir = f"mimic3wdb-matched/1.0/{curr_dir}"

            # Parallel header loading
            with ThreadPoolExecutor(max_workers=8) as executor:
                headers = list(executor.map(
                    lambda hea_file: self.load_header(hea_file[:-4], curr_pn_dir),
                    hea_files
                ))

            valid_pairs = [(idx, header) for idx, header in enumerate(headers) if header is not None]
            if not valid_pairs:
                continue

            complete_set_files = []
            file_durations = []
            file_freqs = []

            for idx, header in valid_pairs:
                sig_names = header.sig_name
                if sig_names is None:
                    continue
                
                # Check PPG + ECG presence
                if not all(label in sig_names for label in ['PLETH', 'II']):
                    continue
                
                # Check ABP/ART presence
                arterial_signals = [label for label in self.Arterial_blood_pressure_labels if label in sig_names]
                if not arterial_signals:
                    continue

                complete_set_files.append(idx)
                file_durations.append(header.sig_len)
                file_freqs.append(header.fs)

            if not complete_set_files:
                continue

            complete_set_files = np.array(complete_set_files)
            file_durations = np.array(file_durations)
            file_freqs = np.array(file_freqs)

            max_idx = np.argmax(file_durations)
            hea_idx = complete_set_files[max_idx]
            max_duration_file = hea_files[hea_idx]
            max_freq = file_freqs[max_idx]
            max_duration = file_durations[max_idx]

            if max_freq > 125:
                print(f"Max Freq: {max_freq:.2f}")

            MIN_LEN = max_freq * 30 * 60
            if max_duration < MIN_LEN:
                continue

            try:
                subj_id = curr_record.split('/')[-2]
                rec_id = max_duration_file[:-4]
                record = wfdb.rdrecord(rec_id, pn_dir=curr_pn_dir, sampto=MIN_LEN)
                sig_names = record.sig_name

                # print(f"File: {curr_record}/{max_duration_file} | subj_id: {subj_id} | Duration: {(max_duration/max_freq)/60:.2f} mins")
                print(f"File: {curr_record}{max_duration_file} | subj_id: {subj_id} | rec_id: {rec_id} | file: {hea_file}| Max Signal Samples: {max_duration} | Fs: {max_freq:.2f}  | Duration (mins): {(max_duration/max_freq)/60:.2f}")
                
                subj_data = {
                    'fix': {
                        'subj_id': subj_id,
                        'rec_id': rec_id,
                        'files': max_duration_file,
                        'af_status': -1
                    },
                    'ppg': {},
                    'ekg': {},
                    'bp': {}
                }

                # Extract signals
                subj_data['ppg']['v'] = record.p_signal[:, sig_names.index('PLETH')]
                subj_data['ppg']['fs'] = record.fs
                subj_data['ppg']['method'] = 'PPG from .hea/.dat'

                subj_data['ekg']['v'] = record.p_signal[:, sig_names.index('II')]
                subj_data['ekg']['fs'] = record.fs
                subj_data['ekg']['method'] = 'ECG from lead II'
                subj_data['ekg']['label'] = 'II'

                blood_pressure_label = next(lab for lab in self.Arterial_blood_pressure_labels if lab in sig_names)
                subj_data['bp']['v'] = record.p_signal[:, sig_names.index(blood_pressure_label)]
                subj_data['bp']['fs'] = record.fs
                subj_data['bp']['method'] = f'{blood_pressure_label} from .hea/.dat'

                subject_data.append(subj_data)
                curr_subject_count += 1
                print(f"Added: {curr_record}/{max_duration_file} | Subjects: {curr_subject_count}/{self.num_subjects}")
                
            except Exception as e:
                print(f"Failed to read record {rec_id}: {e}")
            print(f'Num Subjects {curr_subject_count}/{self.num_subjects}')
        savemat(self.out_path, {'data': subject_data})
        print(f"Saved structured data with {len(subject_data)} entries to {self.out_path}")

if __name__ == "__main__":
    
    config = {
        "paths": {
            "local": {
                "root_folder": os.path.expanduser("~/Documents/Projects/heart_rhythm_analysis/data/raw/mimic_ethnicity_data")
            }
        },
        "ethnicity_extract": False,  # Set True to download and include metadata
        "num_subjects": 10
    }

    collator = MimicEthnicityDatasetCollator(config)
    collator.collate_dataset(load_metadata=False, load_waveforms=True, download_waveforms=True)
