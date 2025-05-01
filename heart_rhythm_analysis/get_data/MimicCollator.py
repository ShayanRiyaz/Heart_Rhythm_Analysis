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

def find_matching_rows(df, columns, match_strings):
    pattern = '|'.join(map(re.escape, match_strings))
    regex = re.compile(pattern, flags=re.IGNORECASE)
    values = df[columns].astype(str).to_numpy()
    match_matrix = np.vectorize(lambda x: bool(regex.search(x)))(values)
    row_matches = match_matrix.any(axis=1)
    return df.index[row_matches].tolist()

class MimicIVCollator():
    def __init__(self, config,verbose = False):
        self.config = config
        self.root_dir = config.get("paths", {}).get("local", {}).get("root_folder",os.getcwd())     # → the value you want, or None if missing
        self.outfile_version = config.get("paths", {}).get("local", {}).get("outfile_version","") 
        self.mimic_num = config.get("mimic_info", {}).get("mimic_num","4")
        self.mimic_path = config.get("mimic_info", {}).get("mimic_path","mimic4wdb/0.1.0/")
        self.ethnicity_extract = config.get("ethnicity_extract", False)
        self.num_subjects =config.get("num_subjects", 200)
        self.bVerbose = verbose
        self.custom_records = config.get("custom_records",{})
        if bool(self.custom_records) is False:
            self.custom_records['name'] = None
            self.custom_records['filenames'] = None
            self.custom_records['label'] = None

        self.ethnicity_data = None

        if self.mimic_num == "4":
            self.mimic_matched_path = 'https://physionet.org/files/mimic-iv-ecg/1.0/'
        self.out_dir = os.path.join(self.root_dir, "downloaded_files")
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.ecg_ppg_labels = ['pleth','ii']
        self.Arterial_blood_pressure_labels = ['abp','art']
        self.categories_of_interest = config.get("substring_to_match",["sinus rhythm"])
        if self.mimic_num != "4":
            self.categories_of_interest = []
        
        if len(self.categories_of_interest) > 1:
            self.categories_of_interest = self.categories_of_interest[0]
            out_filename =  f"{self.outfile_version}_mimic{self.mimic_num}_{self.categories_of_interest.replace(" ","_") if len(self.categories_of_interest) != 0  else ''}_struct.mat"
            self.out_path = os.path.join(self.root_dir,out_filename)
        else:
            out_filename = f"{self.outfile_version}_mimic{self.mimic_num}{self.custom_records['name'] if self.custom_records['name'] is not None else ""}_struct.mat"
            self.out_path = os.path.join(self.root_dir,out_filename )


    def collate_dataset(self, load_metadata=True, load_waveforms=True, download_waveforms=False):
        print(f"\n ~~~ Downloading and collating MIMIC {self.mimic_num} matched waveform subset ~~~")
        self.setup_paths()

        # if load_metadata and self.ethnicity_extract:
        #     self.ensure_metadata_downloaded()
        #     self.load_ethnicity_data()
        # else:
        #     print("Skipping metadata download/extraction.")
        #     self.ethnicity_data = None

        if download_waveforms:
            self.download_dataset()

        match_records = self.prepare_record_list()
        if load_waveforms:
            self.scan_waveform_directory(match_records)

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

    # def ensure_metadata_downloaded(self):
    #     admissions_path = os.path.join(self.out_dir, "ADMISSIONS.csv")
    #     patients_path = os.path.join(self.out_dir, "PATIENTS.csv")

    #     if not os.path.exists(admissions_path):
    #         url = f"{self.mimic_path}/ADMISSIONS.csv.gz"
    #         self.download_and_extract_gz(url, admissions_path)

    #     if not os.path.exists(patients_path):
    #         url = f"{self.mimic_path}/PATIENTS.csv.gz"
    #         self.download_and_extract_gz(url, patients_path)

    # def load_ethnicity_data(self):
    #     print("Loading ethnicity data...")
    #     admissions_path = os.path.join(self.out_dir, "ADMISSIONS.csv")
    #     patients_path = os.path.join(self.out_dir, "PATIENTS.csv")

    #     admissions = pd.read_csv(admissions_path)
    #     patients = pd.read_csv(patients_path)

    #     df = pd.merge(admissions[['SUBJECT_ID', 'ETHNICITY']],
    #                   patients[['SUBJECT_ID', 'GENDER', 'DOB']],
    #                   on='SUBJECT_ID')
    #     self.ethnicity_data = df

    def download_dataset(self):
        output_folder = self.out_dir
        base_url = self.mimic_path
        os.makedirs(output_folder, exist_ok=True)

        response = requests.get(base_url)
        response.raise_for_status()
        html_text = response.text
        hrefs = []
        for line in html_text.splitlines():
            line = line.strip()
            if 'href="' in line:
                start = line.find('href="') + len('href="')
                end = line.find('"', start)
                href = line[start:end]
                hrefs.append(href)

        file_hrefs = [href for href in hrefs if not href.endswith('/')]
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

    def prepare_record_list(self):
        df_categories = {}
        if self.mimic_num == "3":
            records_db = []
            machine_measurments_db = []
        elif self.mimic_num == "4":
            records_db = pd.read_csv(os.path.join(self.out_dir,'record_list.csv'))
            dtype_cols = { 
                16: str, 17: str, 18: str, 19: str, 20: str, 21: str 
            }
            machine_measurments_db = pd.read_csv(os.path.join(self.out_dir,'machine_measurements.csv'),dtype=dtype_cols)
        
        bMatchOverwrite = True
        waveform_dataset_RECORDS = pd.read_csv(f"https://physionet.org/files/{self.mimic_path}/RECORDS",header=None)
        subjects_series = (
            waveform_dataset_RECORDS[0]
            .str.rsplit("/", n=2)  
            .str[1]                 
            .str.lstrip("p")        
        )
        df_categories['matching_records'] = waveform_dataset_RECORDS[0]
        if self.mimic_num == "3":
            if self.custom_records['filenames'] is not None:
                custom_filnames_df = pd.Series(self.custom_records['filenames']).str.split('-', expand=True)
                custom_filnames_df.columns = ['subject_id', 'year', 'month', 'day', 'hour', 'minute']
                subjects_of_interest = custom_filnames_df['subject_id'].str.lstrip("p")
                lookup = {val: idx for idx, val in enumerate(subjects_series)}
                positions = [lookup[x] for x in subjects_of_interest if x in lookup]
                df_categories['matching_records'] = df_categories['matching_records'][positions]
        elif (self.mimic_num == "4") and (len(self.categories_of_interest) > 0):
            df_categories['report_columns'] = [f'report_{i}' for i in range(0,18)]
            flat = machine_measurments_db[df_categories['report_columns']].values.ravel().astype(str)
            all_text= np.unique(np.char.lower(flat))
            filtered_record_list = records_db.merge(machine_measurments_db[['subject_id', 'study_id']],
                                                    on=['subject_id', 'study_id'],
                                                    how='inner')
            for substring in self.categories_of_interest:
                string_exists = any(substring in s for s in all_text)
                if string_exists:
                    row_matches = [s for s in all_text if substring in s]

                substring_path = os.path.join(self.out_dir,f"{substring.replace(" ","_")}_match_idxs.csv")
                if not os.path.exists(substring_path) or (bMatchOverwrite is True):
                    matches = pd.Series(find_matching_rows(machine_measurments_db, df_categories['report_columns'], row_matches))
                    matches.columns = ['substring']
                    matches.to_csv(substring_path,index=False)
                matches = pd.read_csv(substring_path,index_col=None)

                curr_substring_df = filtered_record_list.loc[matches['0']]
                df_categories['curr_substring_measurement_df'] = machine_measurments_db.iloc[matches['0']]
                unique_subjects = (curr_substring_df['subject_id'].astype(str).unique())
                mask = subjects_series.isin(unique_subjects)
                df_categories['matching_records'] = waveform_dataset_RECORDS.loc[mask, 0]

        return df_categories
    
    def scan_waveform_directory(self,df_categories):
        print("Scanning for waveform files...")
        curr_subject_count = 0
        subject_data = []

        matching_records = df_categories['matching_records']
        total_subjects_available = len(matching_records)
        if self.num_subjects > len(matching_records):
                self.num_subjects = len(matching_records)
        for curr_record in matching_records:
            if self.bVerbose: print('\n')
            subj_id_ecg_dataset = int(curr_record.rpartition('/')[0].rpartition('/')[-1][1:])

            if curr_subject_count >= self.num_subjects:
                break

            data_path = os.path.join("https://physionet.org/files/",self.mimic_path,curr_record)
            temp_folder = pd.read_csv(f'{data_path}RECORDS',header=None)[0][0].rpartition('/')[-1]
            hea_file = f"{temp_folder}.hea"
            curr_pn_dir = os.path.join(self.mimic_path,curr_record)
            if self.mimic_num == "4":
                curr_pn_dir = os.path.join(curr_pn_dir,temp_folder)
            try:
                folder_header = wfdb.rdheader(record_name=hea_file[:-4], pn_dir=curr_pn_dir)
            except Exception as e:
                print(f"Failed to load header {curr_pn_dir}/{hea_file[:-4]}: {e}")   
            
            if (hasattr(folder_header, 'seg_name')):
                if ('~' in folder_header.seg_name):
                    indices_to_remove = [i for i, name in enumerate(folder_header.seg_name) if name == '~']
                    folder_header.seg_name = [name for i, name in enumerate(folder_header.seg_name) if i not in indices_to_remove]
                    folder_header.seg_len = np.delete(folder_header.seg_len, indices_to_remove)
                    
                    k = len(folder_header.seg_len)
                    seg_len = np.array(folder_header.seg_len)
                    top_k_indices = np.argsort(-seg_len)[:k]
                else:
                    top_k_indices = [0]
            else:
                pass

            count = 0
            bSkipSubject = False
            for pos in top_k_indices:
                
                count +=1
                if (hasattr(folder_header, 'seg_name')):
                    curr_record_name = folder_header.seg_name[pos]
                else:
                    curr_record_name = folder_header.record_name

                file_header = wfdb.rdheader(record_name=f"{curr_record_name}", pn_dir=curr_pn_dir)
                sig_names = [item.lower() for item in file_header.sig_name]
                arterial_signals = [label.lower() for label in self.Arterial_blood_pressure_labels if label  in sig_names]
                
                max_duration = file_header.sig_len
                max_freq = file_header.fs
                MIN_LEN = int(max_freq * 30 * 60)

                if (max_duration < MIN_LEN) or (max_freq < 20):
                    if self.bVerbose: print(f'Total Subjects Available: {total_subjects_available} | Discarded: {curr_pn_dir}{curr_record_name} [ Duration: {max_duration/max_freq/60:.2f} | Signals: {sig_names}]')
                    bSkipSubject = True
                    break

                if (sig_names is None) or (not all(label.lower() in sig_names for label in self.ecg_ppg_labels)) or (not arterial_signals):
                    if self.bVerbose: print(f'Rejected Record: {curr_pn_dir}{curr_record_name} [ Duration: {max_duration/max_freq/60:.2f} | Signals: {sig_names}]')
                    if pos == top_k_indices[-1]:
                        if self.bVerbose: print(f'Total Subjects Available: {total_subjects_available} | Discarded: {curr_pn_dir}')
                        bSkipSubject = True
                    continue
                break

            if bSkipSubject:
                total_subjects_available -=1
                continue

            try:
                subj_id = curr_record.split('/')[-2]
                rec_id = file_header.record_name
                file_id = hea_file[:-4]
                record = wfdb.rdrecord(rec_id, pn_dir=curr_pn_dir,sampto=MIN_LEN)
                sig_names = [item.lower() for item in record.sig_name]
                
                unique_notes = ""
                if (self.mimic_num == "4") and (len(self.categories_of_interest) > 0):
                    curr_substring_measurement_df = df_categories['curr_substring_measurement_df']
                    filtered_df = curr_substring_measurement_df[curr_substring_measurement_df['subject_id'] == subj_id_ecg_dataset]
                    concatenated_strings = filtered_df[df_categories['report_columns']].fillna('').apply(
                        lambda row: ' '.join(str(cell) for cell in row),
                        axis=1)
                    unique_notes = concatenated_strings.unique().tolist()
            
                subj_data = {
                    'fix': {
                        'subj_id': subj_id,
                        'rec_id': rec_id,
                        'files': file_id,
                        'af_status': self.custom_records['label'] if self.custom_records['label'] is not None else -1,
                        'subject_notes':unique_notes
                    },
                    'ppg': {},'ekg': {},'bp': {}
                }

                subj_data['ppg']['v'] = record.p_signal[:, sig_names.index(self.ecg_ppg_labels[0])]
                subj_data['ppg']['fs'] = record.fs
                subj_data['ppg']['method'] = 'PPG from .hea/.dat'

                subj_data['ekg']['v'] = record.p_signal[:, sig_names.index(self.ecg_ppg_labels[1])]
                subj_data['ekg']['fs'] = 125;#record.fs*
                subj_data['ekg']['method'] = 'ECG from lead II'
                subj_data['ekg']['label'] = 'II'

                blood_pressure_label = next(lab for lab in self.Arterial_blood_pressure_labels if lab in sig_names)
                subj_data['bp']['v'] = record.p_signal[:, sig_names.index(blood_pressure_label)]
                subj_data['bp']['fs'] = record.fs
                subj_data['bp']['method'] = f'{blood_pressure_label} from .hea/.dat'
 
                subject_data.append(subj_data)
                curr_subject_count += 1
                if self.bVerbose: 
                    print(f"File: {curr_record}{file_id} | subj_id: {subj_id} | rec_id: {rec_id} | file: {file_id}| Max Signal Samples: {max_duration} | Fs: {max_freq:.2f}  | Duration (mins): {(max_duration/max_freq)/60:.2f}")
                    print(f"Added: {curr_record}/{rec_id} | Subjects: {curr_subject_count}/{self.num_subjects}")
            except Exception as e:
                print(f"Failed to read record {curr_pn_dir}{rec_id}: {e}")
        savemat(self.out_path, {'data': subject_data})
        if self.bVerbose: print(f"Saved structured data with {len(subject_data)} entries to {self.out_path}")

if __name__ == "__main__":
    # mimic_num = "3"
    for mimic_num in ["3","4"]:
        MIMIC_3_CUSTOM_DATASET = {}
        MIMIC_3_CUSTOM_DATASET['all_label'] = [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1]
        MIMIC_3_CUSTOM_DATASET['all_filename'] = ['p000608-2167-03-09-11-54','p000776-2184-04-30-15-16', 'p000946-2120-05-14-08-08', 'p004490-2151-01-07-12-36', 'p004829-2103-08-30-21-52', 'p075796-2198-07-25-23-40', 'p009526-2113-11-17-02-12', 'p010391-2183-12-25-10-15', 'p013072-2194-01-22-16-13', 'p013136-2133-11-09-16-58', 'p014079-2182-09-24-13-41', 'p015852-2148-05-03-18-39', 'p016684-2188-01-29-00-06', 'p017344-2169-07-17-17-32', 'p019608-2125-02-05-04-57', 'p022954-2136-02-29-17-52', 'p023824-2182-11-27-14-22', 'p025117-2202-03-15-20-28', 'p026377-2111-11-17-16-46', 'p026964-2147-01-11-18-03', 'p029512-2188-02-27-18-10', 'p043613-2185-01-18-23-52', 'p050089-2157-08-23-16-37', 'p050384-2195-01-30-02-21', 'p055204-2132-06-30-09-34', 'p058932-2120-10-13-23-15', 'p062160-2153-10-03-14-49', 'p063039-2157-03-29-13-35', 'p063628-2176-07-02-20-38', 'p068956-2107-04-21-16-05', 'p069339-2133-12-09-21-14', 'p075371-2119-08-22-00-53', 'p077729-2120-08-31-01-03', 'p087275-2108-08-29-12-53', 'p079998-2101-10-21-21-31', 'p081349-2120-02-11-06-35', 'p085866-2178-03-20-17-11', 'p087675-2104-12-05-03-53', 'p089565-2174-05-12-00-07', 'p089964-2154-05-21-14-53', 'p092289-2183-03-17-23-12', 'p092846-2129-12-21-13-12', 'p094847-2112-02-12-19-56', 'p097547-2125-10-21-23-43', 'p099674-2105-06-13-00-07']
        MIMIC_3_CUSTOM_DATASET['af'] = [
            fname
            for lbl, fname in zip(
                MIMIC_3_CUSTOM_DATASET['all_label'],
                MIMIC_3_CUSTOM_DATASET['all_filename']
            )
            if lbl == 1
        ]
        MIMIC_3_CUSTOM_DATASET['nsr'] = [
            fname
            for lbl, fname in zip(
                MIMIC_3_CUSTOM_DATASET['all_label'],
                MIMIC_3_CUSTOM_DATASET['all_filename']
            )
            if lbl == 0
        ]
        
        bCustomRecords = False
        custom_records =     {}
        if mimic_num == "3":
            mimic_path = 'mimic3wdb-matched/1.0/'
            if bCustomRecords:
                custom_records['name']= 'af'
                custom_records['filenames'] = MIMIC_3_CUSTOM_DATASET['af'] 
                custom_records['label'] = 1
                
        elif mimic_num == "4":
            mimic_path = 'mimic4wdb/0.1.0/'#'https://physionet.org/files/mimic-iv-ecg/1.0/'

        config = {
            "paths": {
                "local": {
                    "root_folder": os.path.expanduser(f"~/Documents/Projects/heart_rhythm_analysis/data/raw/mimic{mimic_num}_data"),
                    "outfile_version": ""
                }
            },
            "ethnicity_extract": False,  # Set True to download and include metadata
            "num_subjects": 100,
            "mimic_info": {
                "mimic_num": mimic_num,
                "mimic_path": mimic_path,
            },
            "substring_to_match": [],
            "custom_records": custom_records
            
        }

        collator = MimicIVCollator(config,verbose=True)
        collator.collate_dataset(load_metadata=False, load_waveforms=True, download_waveforms=False)
