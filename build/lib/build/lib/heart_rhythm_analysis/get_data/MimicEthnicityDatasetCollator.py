import os
import requests
import datetime
import numpy as np
import wfdb
from scipy.io import savemat
import pandas as pd

class MimicEthnicityDatasetCollator:
    def __init__(self, root_folder=None):
        """
        Initializes the collator with default or user–specified paths.
        The default root folder should contain subfolders for each subject.
        """
        # -- setup up dicts for paths and settings --
        self.up = {
            'paths': {
                'local': {
                    'root_folder': root_folder or
                        '/Users/shayanriyaz/Documents/Projects/heart_rhythm_analysis/data/raw/'
                        'mimiciii_ppg_ethnicity_beat_detection/',
                },
                'web': {
                    'root_folder': 'https://physionet.org/files/mimic3wdb-matched/1.0/',
                    'records_file': 'RECORDS',
                    # 'admissions_csv_zipped':
                        # 'https://physionet.org/content/mimiciii/1.4/ADMISSIONS.csv.gz',
                },
            },
            'settings': {
                'req_durn': 10 * 60,        # minimum duration (secs)
                'no_subjs_per_ethnicity': 100,
            }
        }

        # derive local filepaths
        lp = self.up['paths']['local']['root_folder']
        self.up['paths']['local'].update({
            'temp_folder':    os.path.join(lp, 'downloaded_files'),
            'records_file':   os.path.join(lp, 'RECORDS'),
            # 'admissions_gz':  os.path.join(lp, 'ADMISSIONS.csv.gz'),
            # 'admissions_csv': os.path.join(lp, 'ADMISSIONS.csv'),
            # 'admissions_pkl': os.path.join(lp, 'admissions.pkl'),
        })

        # ensure folders exist
        os.makedirs(self.up['paths']['local']['temp_folder'], exist_ok=True)
        os.makedirs(lp, exist_ok=True)

        self.check_WFDB_installation()

    def collate_dataset(self):
        """
        Main entry point: downloads and collates MIMIC‐III heartbeat data
        for Black and White subjects into .mat files.
        """
        print("\n ~~~ Downloading and collating MIMIC III matched waveform subset ~~~")
        self.download_required_files()
        subj_ids = self.extract_matched_recs()
        eths     = self.extract_ethnicities(subj_ids)
        df       = self.identify_rel_subj_ids(subj_ids, eths)
        self.save_lists_of_records(df)
        self.download_dataset(df)
        print("\n ~~~ DONE ~~~")

    def check_WFDB_installation(self):
        """
        Checks whether the WFDB package is installed.
        """
        try:
            import wfdb  # noqa: F401
            print("\n    - WFDB for Python detected.")
        except ImportError:
            raise ImportError(
                "This code requires the WFDB Python package. "
                "Install via `pip install wfdb`."
            )

    def download_required_files(self):
        """
        Download RECORDS file and admissions CSV (gzip → CSV → pickle).
        """
        lp = self.up['paths']['local']
        wp = self.up['paths']['web']

        # -- RECORDS --
        if not os.path.exists(lp['records_file']):
            url = wp['root_folder'] + wp['records_file']
            r = requests.get(url); r.raise_for_status()
            with open(lp['records_file'], 'wb') as f:
                f.write(r.content)
            print(" - Downloaded RECORDS")

        # -- Admissions CSV (.gz) --
        # if not os.path.exists(lp['admissions_gz']):
        #     r = requests.get(wp['admissions_csv_zipped'], stream=True)
        #     r.raise_for_status()
        #     with open(lp['admissions_gz'], 'wb') as f:
        #         for chunk in r.iter_content(1<<20):
        #             f.write(chunk)
        #     print(" - Downloaded admissions .gz")

        # -- Unzip to CSV --
        # if not os.path.exists(lp['admissions_csv']):
        #     import gzip, shutil
        #     with gzip.open(lp['admissions_gz'], 'rb') as zin, \
        #          open(lp['admissions_csv'], 'wb') as zout:
        #         shutil.copyfileobj(zin, zout)
        #     print(" - Unzipped admissions CSV")

        # -- Load into pandas & pickle --
        # df = pd.read_csv(
        #     lp['admissions_csv'],
        #     parse_dates=['ADMITTIME','DISCHTIME','DEATHTIME']
        # )
        # df.to_pickle(lp['admissions_pkl'])
        # print(" - Loaded admissions into pickle")

    def extract_matched_recs(self):
        """
        Read the RECORDS file, parse subject IDs, return sorted unique list.
        """
        records_file = self.up['paths']['local']['records_file']
        subj_ids = []
        with open(records_file) as f:
            for line in f:
                sid = int(line.strip()[1:6])  # e.g. 'p000020/...'
                subj_ids.append(sid)
        subj_ids = sorted(set(subj_ids))
        print(f" - Found {len(subj_ids)} matched subject IDs")
        return subj_ids

    # def extract_ethnicities(self, subj_ids):
    #     """
    #     Lookup ETHNICITY for each subject via the admissions pickle.
    #     """
    #     df = pd.read_pickle(self.up['paths']['local']['admissions_pkl'])
    #     df = df[df['SUBJECT_ID'].isin(subj_ids)]
    #     eths = df.set_index('SUBJECT_ID')['ETHNICITY'].to_dict()
    #     print(f" - Extracted ethnicities for {len(eths)} subjects")
    #     return eths

    def identify_rel_subj_ids(self, subj_ids, eths_dict):
        """
        Filter to only White or Black ethnicity, return a pandas DataFrame.
        """
        rows = []
        for sid in subj_ids:
            eth = eths_dict.get(sid, '')
            if 'WHITE' in eth.upper() or 'BLACK' in eth.upper():
                rows.append({'subj_id': sid, 'eth': eth})
        df = pd.DataFrame(rows)
        print(f" - {len(df)} subjects are White or Black")
        return df

    def save_lists_of_records(self, df):
        """
        Write out plain‐text lists of RECORDS paths for each ethnicity.
        """
        lp   = self.up['paths']['local']['root_folder']
        reqd = self.up['settings']['req_durn']
        for eth_label in ['Black', 'White']:
            subset = df[df['eth'].str.contains(eth_label.upper())]
            lines = []
            for sid in subset['subj_id']:
                rec    = f"p{sid:06d}"
                folder = f"p0{str(sid).zfill(6)[1]}"
                lines.append(f"{folder}/{rec}/")
            out_file = os.path.join(lp, f"records_{eth_label}")
            with open(out_file, 'w') as f:
                f.write("\n".join(lines))
            print(f" - Wrote {len(lines)} entries to {out_file}")

    def download_dataset(self, df):
        """
        Download recordings for each subject filtered by ethnicity,
        trim to required duration, and save to .mat files.
        """
        lp      = self.up['paths']['local']['root_folder']
        # derive pb_dir from web root URL
        pb_dir  = self.up['paths']['web']['root_folder'] \
                  .replace('https://physionet.org/files/', '').rstrip('/')
        req_dur = self.up['settings']['req_durn']

        for idx, row in df.iterrows():
            sid = row['subj_id']
            eth = row['eth']
            rec_name = f"p{sid:06d}"
            try:
                # Read header to get sampling frequency and length
                header = wfdb.rdheader(rec_name, pb_dir=pb_dir)
                fs = header.fs
                sig_len = header.sig_len
                if sig_len < req_dur * fs:
                    print(f"    - Skipping {rec_name}: duration {sig_len/fs:.1f}s < required {req_dur}s")
                    continue

                # Read the first req_dur seconds of signal
                sig, fields = wfdb.rdsamp(
                    rec_name,
                    sampfrom=0,
                    sampto=req_dur * fs,
                    pb_dir=pb_dir
                )

                # Save to .mat
                out_file = os.path.join(lp, f"{rec_name}.mat")
                savemat(out_file, {
                    'signal': sig,
                    'fields': fields,
                    'ethnicity': eth
                })
                print(f"    - Saved {out_file}")

            except Exception as e:
                print(f"    - Error downloading {rec_name}: {e}")


if __name__ == '__main__':
    collator = MimicEthnicityDatasetCollator()
    collator.collate_dataset()