from rich import print
from rich.console import Console
import os
import requests
import gzip
import shutil
import pandas as pd
import numpy as np
import wfdb
from scipy.io import savemat
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import glob, re
from pathlib import Path
from typing import Iterator, Optional, Tuple, Dict, Any
from datetime import datetime

session = requests.Session()




def find_matching_rows(df, columns, match_strings):
    pattern = '|'.join(map(re.escape, match_strings))
    regex = re.compile(pattern, flags=re.IGNORECASE)
    values = df[columns].astype(str).to_numpy()
    match_matrix = np.vectorize(lambda x: bool(regex.search(x)))(values)
    row_matches = match_matrix.any(axis=1)
    return df.index[row_matches].tolist()

class MimicCollator():
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
        self.min_minutes = 30
        self.custom_label = config.get("custom_label", -1)
        self.version_num = config.get("version_num", 1)
        if self.mimic_num == "4":
            self.mimic_matched_path = 'https://physionet.org/files/mimic-iv-ecg/1.0/'
        self.out_dir = os.path.join(self.root_dir, "downloaded_files")
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.ecg_ppg_labels = ['pleth','ii']
        self.abp_labels = ['abp','art']
        self.categories_of_interest = config.get("substring_to_match",["sinus rhythm"])
        if self.mimic_num != "4":
            self.categories_of_interest = []
        
        # if len(self.categories_of_interest) > 1:
        #     self.categories_of_interest = self.categories_of_interest[0]
        #     out_filename =  f"{self.outfile_version}_mimic{self.mimic_num}_{self.categories_of_interest.replace(" ","_") if len(self.categories_of_interest) != 0  else ''}_struct.mat"
        #     self.out_path = os.path.join(self.root_dir,out_filename)
        # else:
        #     out_filename = f"{self.outfile_version}_mimic{self.mimic_num}{self.custom_records['name'] if self.custom_records['name'] is not None else ""}_structt.mat"
        #     out_filename = out_filename[1:] if out_filename[0] == '_' else out_filename
        #     self.out_path = os.path.join(self.root_dir,out_filename )


    def collate_dataset(self, load_metadata=True, load_waveforms=True, download_waveforms=False):

        start_time1 = time.time()
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

        if self.mimic_num == "3":
            match_records = self.prepare_mimic3_record_list()
        elif self.mimic_num == "4":
            match_records = self.prepare_mimic4_record_list()

        existing_n,csv_path  = self._find_best_records_file(self.num_subjects)
        if (existing_n is not None) and (self.num_subjects <= existing_n) and csv_path:
            df = pd.read_csv(csv_path)
            if existing_n > self.num_subjects:
                df = df.sample(n=self.num_subjects, random_state=42).reset_index(drop=True)
                print(f"Loaded {existing_n}→sampled to {self.num_subjects} from {csv_path.name}")
            else:
                print(f"Loaded {existing_n} from {csv_path.name}")
        else:
            print("No best‐records CSV found; extracting afresh…")
            df = self.extract_matching_record_info(
                matching_records=match_records['matching_records'],
                mimic_path=self.mimic_path,
                mimic_num=self.mimic_num
            )
            out = Path(self.root_dir) / f"mimic{self.mimic_num}_{self.num_subjects}_best_records.csv"
            df.to_csv(out, index=False)
            print(f"Wrote new {out.name}")

        if load_waveforms:
            self.scan_waveform_directory(match_records,record_info = df)

        end_time1 = time.time()-start_time1
        return end_time1


    def get_rec_id(self,curr_record):
        url = f"https://physionet.org/files/{self.mimic_path}{curr_record}RECORDS"
        resp = session.get(url)
        resp.raise_for_status()
        # grab just the first line, strip trailing slash and path
        first = resp.text.splitlines()[0].rstrip("/")
        return first.split("/")[-1]

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

    def _find_best_records_file(self, desired_n: int) -> Tuple[Optional[Path], int]:
        """
        Return (path, count) of the CSV that:
         - has the smallest count ≥ desired_n, or
         - otherwise the largest count < desired_n,
         - or (None, 0) if none exist.
        """
        pattern = Path(self.root_dir) / f"mimic{self.mimic_num}_*_best_records.csv"
        files = list(pattern.parent.glob(pattern.name))
        if not files:
            return None, 0

        rx = re.compile(rf"mimic{self.mimic_num}_(\d+)_best_records\.csv$")
        supersets, subsets = [], []
        for f in files:
            m = rx.match(f.name)
            if not m:
                continue
            cnt = int(m.group(1))
            (supersets if cnt >= desired_n else subsets).append((cnt, f))

        if supersets:
            return min(supersets, key=lambda x: x[0])
        if subsets:
            return max(subsets, key=lambda x: x[0])
        return None, 0
        
    def extract_matching_record_info(self,matching_records, mimic_path, mimic_num):
        infos = []
        MIN_LEN_FACTOR = self.min_minutes * 60  # minutes → seconds
        cache_path = os.path.join(self.out_dir, f"mimic{self.mimic_num}_records_cache.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                records_cache = json.load(f)
        else:
            records_cache = {}
            for rec in matching_records:
                url = f"https://physionet.org/files/{mimic_path}{rec}RECORDS"
                txt = session.get(url).text
                folder = txt.splitlines()[0].rstrip("/").rsplit("/", 1)[-1]
                records_cache[rec] = folder

            with open(cache_path, "w") as f:
                json.dump(records_cache, f, indent=2)

        time_start1 = time.time()
        if isinstance(matching_records, dict):
            record_list = list(matching_records.values())
        else:
            record_list = list(matching_records)

        for idx, curr_record in enumerate(record_list):
            rec_id = records_cache[curr_record]
            if not rec_id: continue
            
            if len(infos) > self.num_subjects:
                self.num_subjects = len(infos)
                break
            subject_and_rec_dir = os.path.join(mimic_path, curr_record)
            if mimic_num == "3":
                base_pn_dir = subject_and_rec_dir
                initial_header =  rec_id[:-2] if rec_id[-1] == "n" else rec_id
            elif mimic_num == "4":
                base_pn_dir = os.path.join(subject_and_rec_dir, rec_id)
                initial_header = rec_id
            try:
                signal_length_header = wfdb.rdheader(initial_header, pn_dir=base_pn_dir)
            except Exception as e:
                continue

            if (hasattr(signal_length_header, 'seg_name')):
                if ('~' in signal_length_header.seg_name):
                    indices_to_remove = [i for i, name in enumerate(signal_length_header.seg_name) if name == '~']
                    signal_length_header.seg_name = [name for i, name in enumerate(signal_length_header.seg_name) if i not in indices_to_remove]
                    signal_length_header.seg_len = np.delete(signal_length_header.seg_len, indices_to_remove)
                     
            signal_name_file = signal_length_header.seg_name[0]
            if mimic_num == "3":
                if signal_length_header.record_name[-1] == "n":
                    continue
            possible_signals_hdr = wfdb.rdheader(signal_name_file, pn_dir=base_pn_dir)
            
            try:
                possible_signals = [s.lower() for s in possible_signals_hdr.sig_name]
            except:
                possible_signals = []

            seg_names = signal_length_header.seg_name or []
            fs = signal_length_header.fs

            seg_lengths = np.atleast_1d(signal_length_header.seg_len)
            seg_info = list(zip(seg_names, seg_lengths))
            seg_info.sort(key=lambda pair: pair[1], reverse=True)
        
            arterial_signals = [label.lower() for label in self.abp_labels if label  in possible_signals]

            iMinLengthRequired = int(MIN_LEN_FACTOR * 60)

            bNumRecordsCheck = (len(seg_info) < 2)
            bSignalsPresent = (possible_signals is None)
            bEcgPpgPresent = (not all(label.lower() in possible_signals for label in self.ecg_ppg_labels))
            bArterialSignalsPresent = (not arterial_signals)
            if  bNumRecordsCheck or (seg_info[1][1] < iMinLengthRequired) or bSignalsPresent or bEcgPpgPresent or bArterialSignalsPresent:
                continue

            for seg_name, seg_len in seg_info:
                try:
                    seg_hdr = wfdb.rdheader(seg_name, pn_dir=base_pn_dir)
                    actual_signals = [s.lower() for s in seg_hdr.sig_name]
                except:
                    actual_signals = []

                max_duration = seg_len
                max_freq = seg_hdr.fs
                
                arterial_signals = [label.lower() for label in self.abp_labels if label  in actual_signals]
                
                bRecordDurationCheck = (max_duration < iMinLengthRequired)
                bSignalsPresent = (actual_signals is None)
                bMaxFreqCheck = (max_freq < 20)
                bEcgPpgPresent = (not all(label.lower() in actual_signals for label in self.ecg_ppg_labels))
                bArterialSignalsPresent = (not arterial_signals)

                if bRecordDurationCheck or bMaxFreqCheck:
                    break

                if bSignalsPresent or bEcgPpgPresent or bArterialSignalsPresent:
                    continue

                infos.append({
                    "idx":      idx,
                    "pn_dir":   subject_and_rec_dir,
                    "rec_id":   str(rec_id),
                    "file_id":  seg_name,
                    "seg_len":   seg_hdr.sig_len,
                    "max_freq":  max_freq,
                    "signals":   actual_signals
                })
                time_end1 = time.time() - time_start1
                print(f'Time to Store {len(infos)}, Records: {time_end1:.2f}')
                break
        df = pd.DataFrame(infos)
        meta = {
        # when and how
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "min_minutes":     self.min_minutes,
        "ecg_ppg_labels":  self.ecg_ppg_labels,
        "abp_labels":      self.abp_labels,
        "mimic_num":       mimic_num,
        "n_requested":     self.num_subjects,
        "n_returned":      len(df),
        "timestamp_unix":  time.time(),
        "extract_cache":   os.path.basename(cache_path),
        }
        meta_path = os.path.join(self.out_dir, f"mimic{self.mimic_num}_records_metadata.json")
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)
        print(f"📋 Saved metadata → {meta_path}")
        return df

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



    def prepare_mimic3_record_list(self):
        df_categories = {}
        # read the RECORDS file and extract subject series
        recs = pd.read_csv(
            f"https://physionet.org/files/{self.mimic_path}/RECORDS",
            header=None, names=["path"]
        )
        df_categories['matching_records'] = recs["path"]
        subjects_series = recs["path"].str.split("/", expand=True)[1].str.lstrip("p")

        # ── optimize mimic 3 case ───────────────────────────────
        fnames = self.custom_records.get("filenames")
        if fnames:
            # extract subject IDs in one vectorized go
            custom_df = pd.DataFrame(fnames, columns=["filename"])
            custom_df["subject_id"] = (
                custom_df["filename"]
                .str.extract(r"p(\d+)-", expand=False)
            )
            # boolean mask over subjects_series
            mask = subjects_series.isin(custom_df["subject_id"])
            df_categories['matching_records'] = df_categories['matching_records'][mask.values]
        return df_categories

    def prepare_mimic4_record_list(self):
        df_categories = {}
        dtype_cols = {i: str for i in range(16,22)}
        machine_measurements_db = pd.read_csv(os.path.join(self.out_dir,'machine_measurements.csv'),dtype=dtype_cols)

        # read the RECORDS file and extract subject series
        bMatchOverwrite = True
        recs = pd.read_csv(
            f"https://physionet.org/files/{self.mimic_path}/RECORDS",
            header=None, names=["path"]
        )
        df_categories['matching_records'] = recs["path"]
        subjects_series = recs["path"].str.split("/", expand=True)[1].str.lstrip("p")
        # precompute these once
        report_cols = [f"report_{i}" for i in range(18)]
        df_categories['report_columns'] = report_cols

        if self.categories_of_interest:
            for substring in self.categories_of_interest:
                contains_mask = (machine_measurements_db[report_cols].apply(lambda col: col.str.contains(substring, case=False, na=False)).any(axis=1))

                idxs = contains_mask[ contains_mask ].index.to_series()
                path = os.path.join(self.out_dir, f"{substring.replace(' ','_')}_match_idxs.csv")
                if not os.path.exists(path) or bMatchOverwrite:
                    idxs.to_csv(path, index=False, header=False)

                matches = pd.read_csv(path, header=None)[0]

                df_categories['curr_measurement_df'] = machine_measurements_db.loc[matches]

                subject_ids = df_categories['curr_measurement_df']["subject_id"].astype(str).unique()
                mask = subjects_series.isin(subject_ids)
                df_categories['matching_records'] = recs["path"][mask.values]
        else:
            df_categories['curr_measurement_df'] = machine_measurements_db

        return df_categories


    def get_records_iterator(self,
        df_categories: Dict[str, Any],
        record_info: Optional[pd.DataFrame]
    ) -> Iterator[Tuple[int,str]]:
        """Yield (idx, record_path) in the correct order, respecting record_info if given."""
        matching = df_categories["matching_records"]
        if record_info is not None:
            matching = matching.iloc[record_info["idx"]]
        if hasattr(matching, "items"):
            yield from matching.items()
        else:
            yield from enumerate(matching)

    def compute_paths(self,
        curr_record: str,
        curr_idx: int,
        record_info: Optional[pd.DataFrame],
        df_categories: Dict[str, Any],
    ) -> Tuple[str, str, str, str]:
        """
        Returns (pn_dir, hea_file, rec_id, file_id).
        Encapsulates all the logic for figuring out filesystem vs URL paths.
        """
        base = os.path.join(self.mimic_path, curr_record)
        if record_info is not None:
            mask = record_info["idx"] == curr_idx
            rec_id = str(record_info.loc[mask, "rec_id"].iat[0])
            file_id = str(record_info.loc[mask, "file_id"].iat[0])
            hea_file = f"{file_id}.hea"
            if self.mimic_num == "4":
                pn_dir = os.path.join(base, rec_id)
            else:
                pn_dir = base
        else:
            # the old “download then read RECORDS” fallback
            data_url = f"https://physionet.org/files/{self.mimic_path}{curr_record}"
            temp_folder = pd.read_csv(f"{data_url}/RECORDS", header=None).iat[0, 0].rsplit("/", 1)[-1]
            hea_file = f"{temp_folder}.hea"
            pn_dir = os.path.join(base, temp_folder) if self.mimic_num == "4" else base
            rec_id = temp_folder
            file_id = temp_folder
        return pn_dir, hea_file, rec_id, file_id

    def load_folder_header(self,pn_dir: str, hea_file: str):
        """Read the .hea header, stripping out any '~' segments."""
        hdr = wfdb.rdheader(hea_file[:-4], pn_dir=pn_dir)
        if hasattr(hdr, "seg_name") and "~" in hdr.seg_name:
            bad = [i for i,n in enumerate(hdr.seg_name) if n == "~"]
            hdr.seg_name = [n for i,n in enumerate(hdr.seg_name) if i not in bad]
            hdr.seg_len  = np.delete(hdr.seg_len, bad)
        return hdr

    def choose_segment(self,
        folder_hdr,
        pn_dir,
        min_length_s: int,
    ) -> Optional[str]:
        """
        Return the first segment name that meets duration, freq, and signal-presence criteria,
        or None if we should skip this record entirely.
        """
        lengths = np.atleast_1d(folder_hdr.seg_len) if hasattr(folder_hdr, "seg_len") else [folder_hdr.sig_len]
        names   = getattr(folder_hdr, "seg_name", [folder_hdr.record_name])
        order   = np.argsort(lengths)
        for pos in order:
            nm = names[pos]
            fh = wfdb.rdheader(nm, pn_dir=pn_dir)
            if fh.sig_len   < min_length_s:   continue
            if fh.fs        < 20:            continue
            sigs = [s.lower() for s in fh.sig_name]
            if not all(lbl in sigs for lbl in self.ecg_ppg_labels):  continue
            if not any(lbl in sigs for lbl in self.abp_labels):     continue
            return nm
        return None

    def load_waveform(self,
        segment_name: str,
        pn_dir: str,
        sample_to: int
    ):
        """Return the rdrecord object truncated to sample_to."""
        return wfdb.rdrecord(segment_name, pn_dir=pn_dir, sampto=sample_to)

    def extract_notes(self,
        df_categories: Dict[str,Any],
        subj_id_ecg: int
    ) -> Any:
        """Pull out the free-text notes for this subject (MIMIC-4 only)."""
        df = df_categories.get("curr_measurement_df")
        cols = df_categories.get("report_columns", [])
        if df is None or not cols:
            return []
        filt = df["subject_id"] == subj_id_ecg
        texts = df.loc[filt, cols].fillna("").agg(" ".join, axis=1)
        return texts.str.lower().unique().tolist()

    def build_subject_data(self,
        record,
        segment_name: str,
        subj_id: str,
        rec_id: str,
        file_id: str,
        notes: Any,
    ) -> Dict[str,Any]:
        """Compose the dict for this one subject."""
        sigs = [s.lower() for s in record.sig_name]
        ppg_i = sigs.index(self.ecg_ppg_labels[0])
        ecg_i = sigs.index(self.ecg_ppg_labels[1])
        abp_l = next(l for l in self.abp_labels if l in sigs)
        abp_i = sigs.index(abp_l)

        return {
        "fix": {
            "subj_id":     subj_id,
            "rec_id":      rec_id,
            "files":       file_id,
            "af_status":   self.custom_label,
            "subject_notes": notes
        },
        "ppg": {
            "v":      record.p_signal[:, ppg_i],
            "fs":     record.fs,
            "method": "PPG from .hea/.dat"
        },
        "ekg": {
            "v":      record.p_signal[:, ecg_i],
            "fs":     125,
            "method": "ECG from lead II",
            "label":  "II"
        },
        "bp": {
            "v":      record.p_signal[:, abp_i],
            "fs":     record.fs,
            "method": f"{abp_l} from .hea/.dat"
        }
        }

    def scan_waveform_directory(self,
        df_categories: Dict[str,Any],
        record_info: Optional[pd.DataFrame],
    ) -> list:
        subject_data = []
        min_len_s = 60 * self.min_minutes
        for idx, curr_record in self.get_records_iterator(df_categories, record_info):
            if len(subject_data) >= self.num_subjects:
                break

            pn_dir, hea_file, rec_id, file_id = self.compute_paths(
                curr_record,idx, record_info, df_categories)

            try:
                folder_hdr = self.load_folder_header(pn_dir, hea_file)
            except Exception as e:
                if self.bVerbose:
                    print(f"✖ load_folder_header failed ({pn_dir}/{hea_file}): {e}")
                continue

            seg = self.choose_segment(folder_hdr,pn_dir,min_len_s*folder_hdr.fs)
            if not seg:
                if self.bVerbose:
                    print(f"↷ No valid segment in {curr_record}")
                continue

            try:
                record = self.load_waveform(seg, pn_dir, sample_to=min_len_s*folder_hdr.fs)
            except Exception as e:
                if self.bVerbose:
                    print(f"✖ load_waveform failed ({pn_dir}/{seg}): {e}")
                continue

            subj_id = curr_record.split("/")[-2]
            subj_notes = self.extract_notes(df_categories, int(subj_id[1:]))
            data = self.build_subject_data(record, seg, subj_id, rec_id, file_id,subj_notes)
            subject_data.append(data)
            if self.bVerbose:
                print(f"✔ Added {subj_id} ({len(subject_data)}/{self.num_subjects})")
        if subject_data:
            # choose an output filepath
            out_path = getattr(self, 'out_path', None)
            if not out_path:
                # fallback to a sensible default in your root_dir
                fname = f"mimic{self.mimic_num}_data_{self.num_subjects}_v{self.version_num}.mat"
                out_path = os.path.join(self.root_dir, fname)

            # actually write the .mat
            savemat(out_path, {'data': subject_data})
            if self.bVerbose: print(f"Saved {len(subject_data)} subjects → {out_path}")
        else:
            if self.bVerbose: print("No subjects passed all filters; nothing saved.")

        return subject_data

    # def scan_waveform_directory(self,df_categories,record_info=None):
    #     print("Scanning for waveform files...")
    #     curr_subject_count = 0
    #     subject_data = []

    #     matching_records = df_categories['matching_records']
    #     if record_info is not None:
    #         matching_records = matching_records.iloc[record_info['idx']]
            
    #     if hasattr(matching_records, "items"):
    #         iterator = matching_records.items()
    #     else:
    #         iterator = enumerate(matching_records)

    #     if self.num_subjects > len(matching_records):
    #             self.num_subjects = len(matching_records)
    #     total_subjects_available = self.num_subjects
    #     for idx,curr_record in iterator:
    #         if self.bVerbose: print('\n')
    #         subj_id_ecg_dataset = int(curr_record.rpartition('/')[0].rpartition('/')[-1][1:])

    #         if curr_subject_count >= self.num_subjects:
    #             break

    #         curr_pn_dir = os.path.join(self.mimic_path,curr_record)
    #         if record_info is not None:
    #             mask     = record_info['idx'] == idx
    #             rec_id = str(record_info.loc[mask, 'rec_id'].tolist()[0])
    #             file_id = str(record_info.loc[mask, 'file_id'].tolist()[0])
    #             hea_file = f'{file_id}.hea'
    #             if self.mimic_num == "4":
    #                 curr_pn_dir = os.path.join(curr_pn_dir,rec_id)
    #         else:
    #             data_path = os.path.join("https://physionet.org/files/",self.mimic_path,curr_record)
    #             temp_folder = pd.read_csv(f'{data_path}RECORDS',header=None)[0][0].rpartition('/')[-1]
    #             hea_file = f"{temp_folder}.hea"
    #             if self.mimic_num == "4":
    #                 curr_pn_dir = os.path.join(curr_pn_dir,temp_folder)
    #         try:
    #             folder_header = wfdb.rdheader(record_name=hea_file[:-4], pn_dir=curr_pn_dir)
    #         except Exception as e:
    #             print(f"Failed to load header {curr_pn_dir}/{hea_file[:-4]}: {e}")   
    #         top_k_indices = [0]
    #         if (hasattr(folder_header, 'seg_name')):
    #             if ('~' in folder_header.seg_name):
    #                 indices_to_remove = [i for i, name in enumerate(folder_header.seg_name) if name == '~']
    #                 folder_header.seg_name = [name for i, name in enumerate(folder_header.seg_name) if i not in indices_to_remove]
    #                 folder_header.seg_len = np.delete(folder_header.seg_len, indices_to_remove)
                    
    #             k = len(folder_header.seg_len)
    #             seg_len = np.array(folder_header.seg_len)
    #             top_k_indices = np.argsort(-seg_len)[:k]

    #         count = 0
    #         bSkipSubject = False
    #         for pos in top_k_indices:
                
    #             count +=1
    #             if (hasattr(folder_header, 'seg_name')):
    #                 curr_record_name = folder_header.seg_name[pos]
    #             else:
    #                 curr_record_name = folder_header.record_name

    #             file_header = wfdb.rdheader(record_name=f"{curr_record_name}", pn_dir=curr_pn_dir)
    #             sig_names = [item.lower() for item in file_header.sig_name]
    #             arterial_signals = [label.lower() for label in self.Arterial_blood_pressure_labels if label  in sig_names]
                
    #             max_duration = file_header.sig_len
    #             max_freq = file_header.fs
    #             MIN_LEN = int(max_freq * self.min_minutes * 60)

    #             if (max_duration < MIN_LEN) or (max_freq < 20):
    #                 if self.bVerbose: print(f'Total Subjects Available: {total_subjects_available} | Discarded: {curr_pn_dir}{curr_record_name} [ Duration: {max_duration/max_freq/60:.2f} | Signals: {sig_names}]')
    #                 bSkipSubject = True
    #                 break

    #             if (sig_names is None) or (not all(label.lower() in sig_names for label in self.ecg_ppg_labels)) or (not arterial_signals):
    #                 if self.bVerbose: print(f'Rejected Record: {curr_pn_dir}{curr_record_name} [ Duration: {max_duration/max_freq/60:.2f} | Signals: {sig_names}]')
    #                 if pos == top_k_indices[-1]:
    #                     if self.bVerbose: print(f'Total Subjects Available: {total_subjects_available} | Discarded: {curr_pn_dir}')
    #                     bSkipSubject = True
    #                 continue
    #             break

    #         if bSkipSubject:
    #             total_subjects_available -=1
    #             continue

    #         try:
    #             subj_id = curr_record.split('/')[-2]
    #             rec_id = file_header.record_name.split('_')[0]
    #             file_id = hea_file[:-4]
    #             record = wfdb.rdrecord(file_id, pn_dir=curr_pn_dir,sampto=MIN_LEN)
    #             sig_names = [item.lower() for item in record.sig_name]
                
    #             unique_notes = ""
    #             if (self.mimic_num == "4"):
    #                 if (len(self.categories_of_interest) > 0):
    #                     curr_measurement_df = df_categories['curr_measurement_df']
    #                     filtered_df = curr_measurement_df[curr_measurement_df['subject_id'] == subj_id_ecg_dataset]
    #                     concatenated_strings = filtered_df[df_categories['report_columns']].fillna('').apply(
    #                         lambda row: ' '.join(str(cell) for cell in row),
    #                         axis=1)
    #                     unique_notes = concatenated_strings.unique().tolist()
    #                 else:
    #                     subj_df = df_categories["curr_measurement_df"].loc[
    #                     df_categories["curr_measurement_df"]["subject_id"] == int(subj_id[1:]),
    #                     df_categories['report_columns']
    #                     ]
    #                     reports = subj_df.stack().astype(str).str.lower().values
    #                     unique_notes = np.unique(reports)
            
    #             subj_data = {
    #                 'fix': {
    #                     'subj_id': subj_id,
    #                     'rec_id': rec_id,
    #                     'files': file_id,
    #                     'af_status': self.custom_records['label'] if self.custom_records['label'] is not None else -1,
    #                     'subject_notes':unique_notes
    #                 },
    #                 'ppg': {},'ekg': {},'bp': {}
    #             }

    #             subj_data['ppg']['v'] = record.p_signal[:, sig_names.index(self.ecg_ppg_labels[0])]
    #             subj_data['ppg']['fs'] = record.fs
    #             subj_data['ppg']['method'] = 'PPG from .hea/.dat'

    #             subj_data['ekg']['v'] = record.p_signal[:, sig_names.index(self.ecg_ppg_labels[1])]
    #             subj_data['ekg']['fs'] = 125;#record.fs*
    #             subj_data['ekg']['method'] = 'ECG from lead II'
    #             subj_data['ekg']['label'] = 'II'

    #             blood_pressure_label = next(lab for lab in self.Arterial_blood_pressure_labels if lab in sig_names)
    #             subj_data['bp']['v'] = record.p_signal[:, sig_names.index(blood_pressure_label)]
    #             subj_data['bp']['fs'] = record.fs
    #             subj_data['bp']['method'] = f'{blood_pressure_label} from .hea/.dat'
 
    #             subject_data.append(subj_data)
    #             curr_subject_count += 1
    #             print(len(subject_data))
    #             if self.bVerbose: 
    #                 print(f"File: {curr_record}{file_id} | subj_id: {subj_id} | rec_id: {rec_id} | file: {file_id}| Max Signal Samples: {max_duration} | Fs: {max_freq:.2f}  | Duration (mins): {(max_duration/max_freq)/60:.2f}")
    #                 print(f"Added: {curr_record}/{rec_id} | Subjects: {curr_subject_count}/{self.num_subjects}")
    #         except Exception as e:
    #             print(f"Failed to read record {curr_pn_dir}{rec_id}: {e}")
    #     # savemat(self.out_path, {'data': subject_data})
    #     if self.bVerbose: print(f"Saved structured data with {len(subject_data)} entries to {self.out_path}")

if __name__ == "__main__":
    # mimic_num = "3"
    ver_num = 2
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
            mimic_path = 'mimic4wdb/0.1.0/'

        config = {
            "paths": {
                "local": {
                    "root_folder": os.path.expanduser(f"~/Documents/Projects/heart_rhythm_analysis/data/raw/mimic{mimic_num}_data"),
                    "outfile_version": "",
                    "version_num":ver_num
                }
            },
            "ethnicity_extract": False, 
            "num_subjects": 1500,
            "mimic_info": {
                "mimic_num": mimic_num,
                "mimic_path": mimic_path,
            },
            "substring_to_match": [],
            "custom_records": custom_records
        }

        collator = MimicCollator(config,verbose=True)
        end_time = collator.collate_dataset(load_metadata=False, load_waveforms=True, download_waveforms=False)
        print(f"Execution Time: {end_time}")