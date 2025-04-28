from rich import print
from rich.console import Console
import os
import requests
import gzip
import shutil
import re
import pandas as pd
import numpy as np
import wfdb
from scipy.io import savemat
from concurrent.futures import ThreadPoolExecutor


    
def find_matching_rows(df, columns, match_strings):
    # 1. Prepare the regex pattern
    pattern = '|'.join(map(re.escape, match_strings))
    regex = re.compile(pattern, flags=re.IGNORECASE)

    # 2. Extract the columns into a NumPy array
    values = df[columns].astype(str).to_numpy()

    # 3. Flatten matching by vectorized comparison
    match_matrix = np.vectorize(lambda x: bool(regex.search(x)))(values)

    # 4. Find rows where any column matched
    row_matches = match_matrix.any(axis=1)

    # 5. Return matching row indices
    return df.index[row_matches].tolist()


class MimicIVCollator():

    def __init__(self, config):
        self.config = config
        
        self.root_dir = config["paths"]["local"]["root_folder"]
        self.mimic_num = config["mimic_info"]["mimic_num"]
        self.mimic_path = config["mimic_info"]["mimic_path"]
        self.mimic_waveform_url = config["mimic_info"]["mimic_waveform_path"]

        self.out_dir = os.path.join(self.root_dir, "downloaded_files")
        os.makedirs(self.out_dir, exist_ok=True)
        self.ethnicity_extract = config.get("ethnicity_extract", False)
        self.num_subjects =config.get("num_subjects", 200)
        self.ethnicity_data = None
        

        self.ecg_ppg_labels = ['Pleth','II']
        self.Arterial_blood_pressure_labels = ['ABP','ART']
        # self.categories_of_interest = ["fibrillation","sinus rhythm"]
        self.categories_of_interest = ["fibrillation"]
        
        self.out_path = os.path.join(self.root_dir, f"mimic{self.mimic_num}_{self.categories_of_interest[0] if self.categories_of_interest is not None else ''}_struct.mat")
    
    def collate_dataset(self, load_metadata=True, load_waveforms=True, download_waveforms=False):
        print("\n ~~~ Downloading and collating MIMIC III matched waveform subset ~~~")

        self.setup_paths()

        if load_metadata and self.ethnicity_extract:
            self.ensure_metadata_downloaded()
            self.load_ethnicity_data()
        else:
            print("Skipping metadata download/extraction.")
            self.ethnicity_data = None

        if download_waveforms:
            self.download_dataset()
        else:
            records_db = pd.read_csv(os.path.join(self.out_dir,'record_list.csv'))
            records = pd.read_table(os.path.join(self.out_dir,'RECORDS'))
            machine_measurments_db = pd.read_csv(os.path.join(self.out_dir,'machine_measurements.csv'))

        if load_waveforms:
            self.scan_waveform_directory(records,records_db,machine_measurments_db)

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
        admissions_path = os.path.join(self.out_dir, "ADMISSIONS.csv")
        patients_path = os.path.join(self.out_dir, "PATIENTS.csv")

        if not os.path.exists(admissions_path):
            url = f"{self.mimic_path}/ADMISSIONS.csv.gz"
            self.download_and_extract_gz(url, admissions_path)

        if not os.path.exists(patients_path):
            url = f"{self.mimic_path}/PATIENTS.csv.gz"
            self.download_and_extract_gz(url, patients_path)

    def load_ethnicity_data(self):
        print("Loading ethnicity data...")
        admissions_path = os.path.join(self.out_dir, "ADMISSIONS.csv")
        patients_path = os.path.join(self.out_dir, "PATIENTS.csv")

        admissions = pd.read_csv(admissions_path)
        patients = pd.read_csv(patients_path)

        df = pd.merge(admissions[['SUBJECT_ID', 'ETHNICITY']],
                      patients[['SUBJECT_ID', 'GENDER', 'DOB']],
                      on='SUBJECT_ID')
        self.ethnicity_data = df

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

    def download_dataset(self):
        output_folder = self.out_dir
        base_url = self.mimic_path
        os.makedirs(output_folder, exist_ok=True)

        # Step 1: Fetch the page
        response = requests.get(base_url)
        response.raise_for_status()
        html_text = response.text

        # Step 2: Parse manually for <a href="...">
        hrefs = []
        for line in html_text.splitlines():
            line = line.strip()
            if 'href="' in line:
                start = line.find('href="') + len('href="')
                end = line.find('"', start)
                href = line[start:end]
                hrefs.append(href)

        # Step 3: Filter out folders (hrefs ending with '/')
        file_hrefs = [href for href in hrefs if not href.endswith('/')]

        # Step 4: Download files
        for file_href in file_hrefs:
            full_url = base_url + file_href
            local_filename = os.path.join(output_folder, file_href)

            print(f"Downloading: {full_url}")
            file_response = requests.get(full_url, stream=True)
            file_response.raise_for_status()

            with open(local_filename, 'wb') as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Download complete.")
            

    def load_header(self, base_name, pn_dir):
        try:
            return wfdb.rdheader(record_name=base_name, pn_dir=pn_dir)
        except Exception as e:
            print(f"Failed to load header {base_name}: {e}")
            return None
    
    def scan_waveform_directory(self,records,records_db,machine_measurments_db):
        print("Scanning for waveform files...")
        curr_subject_count = 0
        subject_data = []
        filtered_record_list = records_db.merge(
        machine_measurments_db[['subject_id', 'study_id']],
        on=['subject_id', 'study_id'],
        how='inner'
        )

        report_columns = [f'report_{i}' for i in range(0,18)]
        all_text =pd.unique(machine_measurments_db[report_columns].values.ravel()).astype(str)

        for substring in self.categories_of_interest:
            string_exists = any(substring in s for s in all_text)
            if string_exists:
                row_matches = [s for s in all_text if substring in s]

            substring_path = os.path.join(self.out_dir,f"{substring}_match_idxs.csv")
            if not os.path.exists(substring_path):
                matches = find_matching_rows(machine_measurments_db, report_columns, row_matches)
                matches = pd.Series(matches)
                matches.columns = ['substring']
                matches.to_csv(substring_path,index=False)
            else:
                matches = pd.read_csv(substring_path,index_col=None)

            curr_substring_df = filtered_record_list.loc[matches['0']]
            curr_substring_measurement_df = machine_measurments_db.iloc[matches['0']]

            unique_subjects = pd.unique(curr_substring_df['subject_id'].values.ravel()).astype(str)

            waveform_dataset_RECORDS = pd.read_csv("https://physionet.org/files/mimic4wdb/0.1.0/RECORDS",header=None)
            waveform_subject_ids = waveform_dataset_RECORDS[0].str.rpartition('/')[0].str.rpartition('/')[2]
            waveform_subject_ids = waveform_subject_ids.str.replace('p', '', regex=False).tolist()
            
            lookup = {value: idx for idx, value in enumerate(waveform_subject_ids)}
            matching_indices = [lookup[val] for val in unique_subjects if val in lookup]
            matching_records = waveform_dataset_RECORDS.iloc[matching_indices]

        for curr_record in matching_records[0]:
            # subj_id_waveform_dataset = waveform_dataset_RECORDS[waveform_dataset_RECORDS[0].str.contains(curr_record, case=False, na=False)]
            subj_id_ecg_dataset = int(curr_record.rpartition('/')[0].rpartition('/')[-1][1:])
            if self.num_subjects > len(matching_records[0]):
                self.num_subjects = len(matching_records[0])

            if curr_subject_count >= self.num_subjects:
                break

            data_path = os.path.join(self.mimic_waveform_url,curr_record)
            temp_folder = pd.read_csv(f'{data_path}/RECORDS',header=None)[0][0].rpartition('/')[-1]
            hea_file = f"{temp_folder}.hea"
            curr_pn_dir = os.path.join('mimic4wdb/0.1.0/',curr_record,temp_folder)
            
            try:
                folder_header = wfdb.rdheader(record_name=hea_file[:-4], pn_dir=curr_pn_dir)
            except Exception as e:
                print(f"Failed to load header {curr_pn_dir}/{hea_file[:-4]}: {e}")   

            k = len(folder_header.seg_len)
            seg_len = np.array(folder_header.seg_len)
            top_k_indices = np.argsort(-seg_len)[:k]


            # for rec_id,current_subject_record in zip(find_subject_files['file_name'],find_subject_files['path']):
            for pos in top_k_indices:
                curr_record_name = folder_header.seg_name[pos]
                file_header = wfdb.rdheader(record_name=f"{curr_record_name}", pn_dir=curr_pn_dir)
                sig_names = file_header.sig_name
                
                if sig_names is None:
                    continue
                
                # Check PPG + ECG presence
                if not all(label in sig_names for label in self.ecg_ppg_labels):
                    continue
                
                # Check ABP/ART presence
                # arterial_signals = [label for label in self.Arterial_blood_pressure_labels if label in sig_names]
                # if not arterial_signals:
                #     continue
                break
            
            max_duration = file_header.sig_len
            max_freq = file_header.fs

            if max_freq > 65:
                print(f"Max Freq: {max_freq:.2f}")

            MIN_LEN = int(max_freq * 30 * 60)
            if max_duration < MIN_LEN:
                continue

            try:
                subj_id = curr_record.split('/')[-2]
                rec_id = file_header.record_name
                file_id = hea_file[:-4]
                record = wfdb.rdrecord(rec_id, pn_dir=curr_pn_dir,sampto=MIN_LEN)
                sig_names = record.sig_name


                filtered_df = curr_substring_measurement_df[curr_substring_measurement_df['subject_id'] == subj_id_ecg_dataset]
                concatenated_strings = filtered_df[report_columns].fillna('').apply(
                    lambda row: ' '.join(str(cell) for cell in row),
                    axis=1)
                unique_strings = concatenated_strings.unique().tolist()
                
                subj_data = {
                    'fix': {
                        'subj_id': subj_id,
                        'rec_id': rec_id,
                        'files': file_id,
                        'af_status': -1,
                        'subject_notes':unique_strings
                    },
                    'ppg': {},
                    'ekg': {},
                    'bp': {}
                }

                # Extract signals
                subj_data['ppg']['v'] = record.p_signal[:, sig_names.index(self.ecg_ppg_labels[0])]
                subj_data['ppg']['fs'] = record.fs
                subj_data['ppg']['method'] = 'PPG from .hea/.dat'

                subj_data['ekg']['v'] = record.p_signal[:, sig_names.index(self.ecg_ppg_labels[1])]
                subj_data['ekg']['fs'] = record.fs
                subj_data['ekg']['method'] = 'ECG from lead II'
                subj_data['ekg']['label'] = 'II'

                # blood_pressure_label = next(lab for lab in self.Arterial_blood_pressure_labels if lab in sig_names)
                # subj_data['bp']['v'] = record.p_signal[:, sig_names.index(blood_pressure_label)]
                # subj_data['bp']['fs'] = record.fs
                # subj_data['bp']['method'] = f'{blood_pressure_label} from .hea/.dat'

                subject_data.append(subj_data)
                curr_subject_count += 1
                print(f"Added: {curr_record}/{rec_id} | Subjects: {curr_subject_count}/{self.num_subjects}")
                
            except Exception as e:
                print(f"Failed to read record {rec_id}: {e}")
            print(f'Num Subjects {curr_subject_count}/{self.num_subjects}')
        savemat(self.out_path, {'data': subject_data})
        print(f"Saved structured data with {len(subject_data)} entries to {self.out_path}")

if __name__ == "__main__":
    
    mimc_num = "4"
    mimc3_path = 'https://physionet.org/static/published-projects/mimic3wdb-matched/1.0/RECORDS'
    mimc4_path = 'https://physionet.org/files/mimic-iv-ecg/1.0/'
    mimic4_waveform_path = 'https://physionet.org/files/mimic4wdb/0.1.0/'
    config = {
        "paths": {
            "local": {
                "root_folder": os.path.expanduser(f"~/Documents/Projects/heart_rhythm_analysis/data/raw/mimic{mimc_num}_data")
            }
        },
        "ethnicity_extract": False,  # Set True to download and include metadata
        "num_subjects": 10000,
        "mimic_info": {
            "mimic_num": "4",
            "mimic_path": mimc4_path,
            "mimic_waveform_path": mimic4_waveform_path
        }
    }

    collator = MimicIVCollator(config)
    collator.collate_dataset(load_metadata=False, load_waveforms=True, download_waveforms=False)
