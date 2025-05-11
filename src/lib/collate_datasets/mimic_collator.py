'''
MimicCollator.py

Utilities for collating MIMIC III and IV waveform records into matfile sets.

Usage:
    from mimic_collator import MimicCollator
    collator = MimicCollator(config)
    collator.scan_waveform_directory(...)

Author: Shayan Riyaz
License: MIT

'''

from rich import print
import os,time,json, re, requests,wfdb
from scipy.io import savemat
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
from urllib.request import urlretrieve

from lib.helper.get_typecast import *

session = requests.Session()

def concat_records(group,record_cols):
    data = group[record_cols].values.flatten()
    # drop nan/empty
    kept = [str(x) for x in data if pd.notna(x) and str(x).strip()!='']
    return ";".join(kept)

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
        self.bVerbose = verbose

        self.version_num = get_typecast(config, "version_num",cast_fn=int, default=0,verbose=self.bVerbose)

        self.root_dir  = get_typecast(config, "paths","local","root_folder",cast_fn=Path, default=os.getcwd(),verbose=self.bVerbose)
        self.mimic_path  = get_typecast(config, "mimic_info","mimic_path",cast_fn=str, default="mimic4wdb/0.1.0/",verbose=self.bVerbose)

        self.out_dir = self.root_dir / "downloaded_files"

        self.mimic_num = get_typecast(config, "mimic_info","mimic_num", cast_fn=str, default="4",verbose=self.bVerbose)
        self.ethnicity_extract = config.get("ethnicity_extract", False)
        self.custom_records = get_typecast(config, "custom_records", cast_fn=lambda v: pd.DataFrame(v), default=pd.DataFrame(),verbose=self.bVerbose)
        self.ethnicity_data = None

        self.num_subjects = self.remaining_subjects = get_typecast(config, "num_subjects", cast_fn=int, default=50,verbose=self.bVerbose)
        self.MAX_SET_SIZE = get_typecast(config, "max_set_size", cast_fn=int, default=50,verbose=self.bVerbose)
        self.min_freq = get_typecast(config, "min_freq", cast_fn=int, default=60,verbose=self.bVerbose)
        self.start_minutes = 5
        self.end_minutes = get_typecast(config, "min_minutes", cast_fn=int, default=60,verbose=self.bVerbose)

        self.start_samples = int(60*self.start_minutes*self.min_freq)
        self.end_samples = int(( self.end_minutes*60*self.min_freq)+ self.start_samples)

        self.custom_label = config.get("custom_label", -1)    
        self.ecg_labels = [sig.lower() for sig in config.get("ecg_labels", ['ii'])]
        self.ppg_labels = [sig.lower() for sig in config.get("ppg_labels", ['pleth'])]
        self.abp_labels = [sig.lower() for sig in config.get("abp_labels", ['abp','art'])]

        self.categories_of_interest = None
        if self.mimic_num == "4":
            self.mimic_matched_path = 'https://physionet.org/files/mimic-iv-ecg/1.0/'
        

    def collate_dataset(self, load_metadata=True, load_waveforms=True, download_waveforms=False):
        ''' 
        -> COLLATE_DATASET
            - Inputs:

            - Ouputs:
        '''
        print(f"\n --- Downloading and collating MIMIC {self.mimic_num} matched waveform subset ---")
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

        if self.num_subjects > match_records.shape[0]:
            if self.bVerbose: print(f'Not enough subjects available: Setting num_subjects from {self.num_subjects} to {match_records.shape[0]}')
            self.num_subjects = match_records.shape[0]

        existing_n,csv_path  = self._find_best_records_file(self.num_subjects)
        if (existing_n is not None) and (self.num_subjects <= existing_n) and csv_path:
            df = pd.read_csv(csv_path)
            if existing_n > self.num_subjects:
                df = df.sample(n=self.num_subjects, random_state=42).reset_index(drop=True)
                print(f"Loaded {existing_n}→sampled to {self.num_subjects} from {csv_path.name}")
            else:
                print(f"Loaded {existing_n} from {csv_path.name}")
        else:
            print("No best‐records CSV found; extracting new mapping")
            df = self.extract_matching_record_info(match_records)
            
            out = Path(self.root_dir) / f"mimic{self.mimic_num}_{df.shape[0]}_best_records.csv"
            df.to_csv(out, index=False)
            print(f"Wrote new {out.name}")

        if self.num_subjects > df.shape[0]:
            self.num_subjects = self.remaining_subjects = df.shape[0]
        if load_waveforms:
            self.scan_waveform_directory(match_records,record_info = df)
        return 
    




    def get_rec_id(self,curr_record):
        url = f"https://physionet.org/files/{self.mimic_path}{curr_record}RECORDS"
        resp = session.get(url)
        resp.raise_for_status()
        first = resp.text.splitlines()[0].rstrip("/")
        return first.split("/")[-1]
    




    def download_dataset(self,base_url):
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
            full_url = self.mimic_path + file_href
            local_filename = os.path.join(self.out_dir, file_href)

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
    




    def signal_requirements_check(self,filename,pn_dir,header_type):
        """Read the .hea header, stripping out any '~' segments."""
        try:
            hdr = wfdb.rdheader(filename, pn_dir=pn_dir)
        except Exception as e:
            return None,0,None
            
        if header_type == 'folder':
            if hasattr(hdr, "seg_name") and "~" in hdr.seg_name:
                bad = [i for i,n in enumerate(hdr.seg_name) if n == "~"]
                hdr.seg_name = [n for i,n in enumerate(hdr.seg_name) if i not in bad]
                hdr.seg_len  = np.delete(hdr.seg_len, bad)
            return hdr,1,None
        
        try:
            actual_signals = [s.lower() for s in hdr.sig_name]
        except:
            actual_signals = None
            return hdr,0,actual_signals
        
        if header_type == 'file':
            if (hdr.fs < self.min_freq):
                return hdr,0,actual_signals
            
        if self.abp_labels is not None:
            arterial_signals_present = [label.lower() for label in self.abp_labels if label  in actual_signals]
            if not arterial_signals_present: 
                return hdr,0,actual_signals

        if self.ecg_labels is not None:
            ecg_labels_present = all(label.lower() in actual_signals for label in self.ecg_labels)
            if not ecg_labels_present: 
                return hdr,0,actual_signals

        if self.ppg_labels is not None:
            ppg_labels_present = all(label.lower() in actual_signals for label in self.ppg_labels)
            if not ppg_labels_present: 
                return hdr,0,actual_signals

        return hdr,1,actual_signals
    




    def setup_paths(self):
        print("\n - Setting up parameters")
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Root folder does not exist: {self.root_dir}")
        print("Working directory verified:", self.root_dir)





    def prepare_mimic3_record_list(self):
        df_categories = {}
        # read the RECORDS file and extract subject series
        recs = pd.read_csv(f"https://physionet.org/files/{self.mimic_path}/RECORDS-waveforms",header=None, names=["path"])

        # df_categories['matching_records'] = recs["path"]
        # subjects_series = recs["path"].str.split("/", expand=True)[1].str.lstrip("p")
        df_categories = pd.DataFrame({
            'matching_records':recs["path"].str.rpartition('/')[0] + '/',
            'subject_id': recs["path"].str.split("/", expand=True)[1].str.lstrip("p"),
            'filename': recs["path"].str.split("/", expand=True)[2]
        })
        df_categories = df_categories.drop_duplicates(subset='subject_id', keep='first')

        if not self.custom_records.empty: 
            ids_in_df2 = set(self.custom_records["subject_id"])
            df_categories['in_df2'] = df_categories['subject_id'].isin(ids_in_df2)
            any_matches = df_categories['in_df2'].any()
            df_categories['notes'] = str()
            if any_matches:
                label_map = self.custom_records.set_index('subject_id')['label'].to_dict()
                df_categories['notes'] = df_categories['subject_id'].map(label_map).fillna("").astype(str)
        df_categories.reset_index(inplace=True,drop=True)
        return df_categories
    




    def prepare_mimic4_record_list(self):
        df_categories = {}
        dtype_cols = {i: str for i in range(16,22)}
        machine_measurments_path = self.out_dir / 'machine_measurements.csv'
        url      = f"{self.mimic_matched_path}/machine_measurements.csv"

        if not machine_measurments_path.exists():
            machine_measurments_path.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(url, machine_measurments_path)   # C-optimized download
        machine_measurements_db = pd.read_csv(machine_measurments_path, dtype=dtype_cols)
        recs = pd.read_csv(f"https://physionet.org/files/{self.mimic_path}/RECORDS",header=None, names=["path"])
        

        df_categories = pd.DataFrame({
            'matching_records':recs["path"],
            'subject_id':recs["path"].str.split("/", expand=True)[2].str.lstrip("p")
        })

        record_cols = [f"report_{i}" for i in range(18)]  

        def concat_records(group):
            data = group[record_cols].values.flatten()
            # drop nan/empty
            kept = [str(x) for x in data if pd.notna(x) and str(x).strip()!='']
            return ";".join(kept)
        
        ids_in_df2 = set(machine_measurements_db['subject_id'])
        df_categories['in_df2'] = df_categories['subject_id'].isin(ids_in_df2)
        any_matches = df_categories['in_df2'].any()
        df_categories["notes"] = ""
        if any_matches:
            mapping = (machine_measurements_db.groupby("subject_id").apply(concat_records).rename("all_records"))
            df_categories = df_categories[df_categories["subject_id"].isin(mapping.index)].copy()

            df_categories["all_records"] = df_categories["subject_id"].map(mapping)
        return df_categories
    




    def _extract_one_record(self, idx, curr_record, records_cache, df_categories):
        """
        The logic you had inside your for‐loop, reduced to a function that
        takes (idx, curr_record) and either returns an info‐dict or None.
        """
        rec_id = records_cache.get(curr_record)
        if not rec_id:
            return None

        subject_and_rec_dir = os.path.join(self.mimic_path, curr_record)
        if self.mimic_num == "3":
            base_pn_dir     = subject_and_rec_dir
            initial_header  = rec_id[:-2] if rec_id.endswith("n") else rec_id
        else:
            base_pn_dir = os.path.join(subject_and_rec_dir, rec_id)
            initial_header  = rec_id

        if self.mimic_num == "3" and initial_header.endswith("n"):
            return None

        try:
            hdr1, ok1, _ = self.signal_requirements_check(filename=initial_header,pn_dir=base_pn_dir,header_type="folder")
        except:
            return None
        if not ok1:
            return None

        segs = list(zip(hdr1.seg_name, hdr1.seg_len))
        segs.sort(key=lambda x: x[1], reverse=True)
        if len(segs) < 2 or segs[0][1] < int(self.end_samples):
            return None

        # second requirements check
        try:
            _, ok2, actual_signals = self.signal_requirements_check(filename=segs[0][0].split(".")[0],pn_dir=base_pn_dir,header_type="folder")
        except:
            return None
        if not ok2:
            return None

        # finally loop segments until one passes file‐type check
        for seg_name, seg_len in segs:
            if seg_len < int(self.end_samples):
                break
            try:
                hdr3, ok3, actual_signals = self.signal_requirements_check(filename=seg_name,pn_dir=base_pn_dir,header_type="file")
            except:
                continue
            if not ok3:
                continue

            # grab the notes
            notes = df_categories.loc[df_categories["matching_records"] == curr_record,"notes"].squeeze() or ""
            if self.bVerbose: print(f'Added Subject:  idx: {idx}, pn_dir: {base_pn_dir}, rec_id: {rec_id}, file_id: {seg_name}, actual_signals: {actual_signals},\t\tnotes: {notes}')
                    
            return {
                "idx":      idx,
                "pn_dir":   subject_and_rec_dir,
                "rec_id":   str(rec_id),
                "file_id":  seg_name,
                "seg_len":  hdr3.sig_len,
                "max_freq": hdr3.fs,
                "signals":  actual_signals,
                "notes":    notes
            }
        return None
    




    def extract_matching_record_info(self, df_categories, max_workers=8):
        matching_records = df_categories['matching_records']
        cache_path = os.path.join(self.root_dir, f"mimic{self.mimic_num}_records_cache.json")

        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                records_cache = json.load(f) 
        else:
            if self.mimic_num == "3":
                f = open(cache_path, "w")
                    
                records_cache = df_categories.set_index('matching_records').to_dict()['filename']
                json.dump(records_cache, f, indent = 2)
            elif self.mimic_num == "4":
                records_cache = {}
                for rec in matching_records:
                    url = f"https://physionet.org/files/{self.mimic_path}{rec}RECORDS"
                    txt = session.get(url).text
                    folder = txt.splitlines()[0].rstrip("/").rsplit("/", 1)[-1]
                    records_cache[rec] = folder
                with open(cache_path, "w") as f:
                    json.dump(records_cache, f, indent=2)

        # if isinstance(matching_records, dict):
        #     record_list = list(matching_records.values())
        # else:
        #     record_list = list(matching_records)
        record_list = list(records_cache.keys())

        # build args for each record
        args = [(idx, rec, records_cache, df_categories) for idx, rec in enumerate(record_list)]

        infos = []
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(self._extract_one_record, *arg): arg
                for arg in args
            }
            for fut in as_completed(futures):
                info = fut.result()
                if info is not None:
                    infos.append(info)
                    if len(infos) >= self.num_subjects:
                        break

        df = pd.DataFrame(infos)
        df = df.sort_values(by=['idx'], ascending=True)

        meta = {
            "generated_at":  datetime.now(timezone.utc).isoformat() + "Z",
            "start_seconds": self.start_samples,
            "end_seconds":   self.end_samples,
            "ppg_labels":    self.ppg_labels,
            "ecg_labels":    self.ecg_labels,
            "abp_labels":    self.abp_labels,
            "mimic_num":     self.mimic_num,
            "n_requested":   self.num_subjects,
            "n_returned":    len(df),
            "timestamp_unix": time.time(),
            "extract_cache": os.path.basename(cache_path),
        }
        meta_path = os.path.join(
            self.root_dir,
            f"mimic{self.mimic_num}_{df.shape[0]}_best_records_metadata.json"
        )
        with open(meta_path, "w") as mf:
            json.dump(meta, mf, indent=2)
        if self.bVerbose:
            print(f"Saved metadata → {meta_path}")
        return df
    def get_records_iterator(self,df_categories: Dict[str, Any],
        record_info: Optional[pd.DataFrame]) -> Iterator[Tuple[int,str]]:
        """
        Yield (idx, record_path) in the correct order, respecting record_info if given.
        """
        matching = df_categories["matching_records"]
        if record_info is not None:
            matching = matching.iloc[record_info["idx"]]
        if hasattr(matching, "items"):
            yield from matching.items()
        else:
            yield from enumerate(matching)





    def compute_paths(self,curr_record: str,
                    curr_idx: int,
                    record_info: Optional[pd.DataFrame]) -> Tuple[str, str, str, str]:
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
    




    def load_waveform(self,
        segment_name: str,
        pn_dir: str,
        sampfrom: int,
        sampleto: int,
        channels: list
    ):
        """Return the rdrecord object truncated to sample_to."""
        return wfdb.rdrecord(segment_name, pn_dir=pn_dir,sampfrom=sampfrom, sampto=sampleto,channels = channels,return_res=32)
    


    def extract_notes(self,df_categories,subj_id_ecg
    ) -> Any:
        """Pull out the free-text notes for this subject (MIMIC-4 only)."""
        df = df_categories.get("curr_measurement_df")
        cols = df_categories.get("report_columns", [])
        if df is None or not cols:
            return [""]
        filt = df["subject_id"] == subj_id_ecg
        texts = df.loc[filt, cols].fillna("").agg(" ".join, axis=1)
        if texts.empty: 
            return [""]
        return texts.str.lower().unique().tolist()
    




    def build_subject_data(self,
        record,
        subj_id: str,
        rec_id: str,
        file_id: str,
        notes: Any,
    ) -> Dict[str,Any]:
        """Compose the dict for this one subject."""
        sigs = [s.lower() for s in record.sig_name]
        abp_l = next(l for l in self.abp_labels if l in sigs)

        return {
        "fix": {
            "subj_id":     subj_id,
            "rec_id":      rec_id,
            "files":       file_id,
            "af_status":   self.custom_label,
            "subject_notes": notes
        },
        "ppg": {
            "v":      record.p_signal[:, 0],
            "fs":     record.fs,
            "method": "pleth from .hea/.dat",
            "label" : "pleth"
        },
        "ekg": {
            "v":      record.p_signal[:, 1],
            "fs":     record.fs,
            "method": "ECG from lead II",
            "label":  "ii"
        },
        "bp": {
            "v":      record.p_signal[:, 2],
            "fs":     record.fs,
            "method": f"{abp_l} from .hea/.dat",
            "label":  abp_l 
            }
        }

    def _process_single_record(self, args):
        """
        Worker for one (idx, curr_record) tuple.
        Returns a dict of subject_data or None on failure / skip.
        """
        idx, curr_record, df_categories, record_info = args
        record_info = record_info.sort_values(by=['idx'], ascending=True)
        try:
            pn_dir, hea_file, rec_id, file_id = self.compute_paths(
                curr_record, idx, record_info
            )

            file_header,pass_check,sigs = self.signal_requirements_check(filename=hea_file.split('.')[0],pn_dir=pn_dir,header_type='file')
            seg_name = file_header.record_name

            if pass_check == 0:
                return data
            ppg_i = [i for i, sig in enumerate(sigs) if sig.lower() in self.ppg_labels]
            ecg_i = [i for i, sig in enumerate(sigs) if sig.lower() in self.ecg_labels]
            abp_l = next(l for l in self.abp_labels if l in sigs)
            abp_i = sigs.index(abp_l)
            channels = [ppg_i,ecg_i,abp_i]
            # channels = [i for sub in channels for i in (sub if isinstance(sub, (list, tuple)) else [sub])]
            channels = [x for sub in channels for x in (sub if isinstance(sub, list) else [sub])]
            record = self.load_waveform(
                seg_name, pn_dir,
                sampfrom=int(self.start_minutes*60*file_header.fs),
                sampleto=int(self.end_minutes*60*file_header.fs),
                channels=channels)

            subj_id     = curr_record.split("/")[-2]
            subj_notes  = self.extract_notes(df_categories, int(subj_id[1:]))
            data        = self.build_subject_data( record, subj_id, rec_id, file_id, subj_notes)
            return data

        except Exception as e:
            if self.bVerbose:
                print(f"Skipping {curr_record!r}: {e}")
            return None

    def scan_waveform_directory(self,df_categories: Dict[str,Any],
                                record_info: Optional[pd.DataFrame],
                                max_workers:int = 1) -> list:
        """
        Parallelized scan over get_records_iterator.
        """

        subject_data      = []
        set_num           = 0
        num_subjects_added = 0
        num_remaining_subjects = self.num_subjects
        arg_iter = [
            (idx, rec, df_categories, record_info)
            for idx, rec in self.get_records_iterator(df_categories, record_info)
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(self._process_single_record, args): args for args in arg_iter}        
            for fut in as_completed(futures):
                if num_subjects_added >= self.num_subjects:
                    break

                data = fut.result()
                if data is None:
                    continue

                subject_data.append(data)
                num_subjects_added += 1
                num_remaining_subjects -= 1
                if self.bVerbose: print(f"Remainings Subject ({num_remaining_subjects}/{self.num_subjects-set_num*self.MAX_SET_SIZE}) | # Subjects Added in Set: {num_subjects_added}")
                
                if (num_subjects_added >= self.MAX_SET_SIZE) or (num_remaining_subjects == 0):
                    set_num += 1
                    folder_name = 'subject_sets'
                    fname = f"mimic{self.mimic_num}_data_{self.num_subjects}_{set_num}_v{self.version_num}.h5"
                    out_dir = os.path.join(self.root_dir, folder_name)
                    out_path = os.path.join(out_dir, fname)
                    if not os.path.exists(out_dir):
                        os.mkdir(out_dir)
                    savemat(out_path, {'data': subject_data})
                    if self.bVerbose: print(f"Saved {len(subject_data)} Subjects → {fname}")
                    subject_data = []
                    num_subjects_added = 0
        return subject_data
    

    # def download_and_extract_gz(self, url, out_path):
    #     print(f"Downloading from {url}")
    #     gz_path = out_path + ".gz"
    #     with requests.get(url, stream=True) as r:
    #         r.raise_for_status()
    #         with open(gz_path, 'wb') as f:
    #             shutil.copyfileobj(r.raw, f)
    #     with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
    #         shutil.copyfileobj(f_in, f_out)
    #     os.remove(gz_path)

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

if __name__ == "__main__":
    

    VERSION_NUM = 3
    MIN_MINUTES = 30

    for MIMIC_NUM in ["4"]:
        root_folder = f'data/raw/mimic{MIMIC_NUM}_data/'
        if not os.path.exists(root_folder):
            os.makedirs(root_folder, exist_ok = False)

        custom_records = {}
        bCustomRecords = True
        if MIMIC_NUM == "3":
            mimic_path = 'mimic3wdb-matched/1.0/'
            MAX_SUBJECTS = 2100
            if bCustomRecords: 
                try:
                    custom_records = pd.read_csv(f"{root_folder}/mimic3_annotations.csv",header = 0,index_col=0,dtype=str)
                except Exception as e:
                    print(f'{e}, run "python src/lib/collate_datasets/create_mimic3_notes.py" in terminal to create the file')
        elif MIMIC_NUM == "4":
            mimic_path = 'mimic4wdb/0.1.0/'
            MAX_SUBJECTS = 53

        config = {
            "paths": {
                "local": {
                 "root_folder": Path(root_folder),
                },
            },
            "mimic_info": {
                "mimic_num": MIMIC_NUM,
                "mimic_path": mimic_path,
            },
            "version_num":VERSION_NUM,
            "min_minutes" : MIN_MINUTES,
            "num_subjects": MAX_SUBJECTS,
            "categories_of_interest": [],
            "custom_records": custom_records,
            "ethnicity_extract": False
        }

        collator = MimicCollator(config,verbose=True)
        collator.collate_dataset(load_metadata=False, load_waveforms=True, download_waveforms=False)