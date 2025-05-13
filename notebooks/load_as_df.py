import pandas as pd
import h5py
import numpy as np

def load_as_df(file_path, filename,
               store_signals=True,
               group_by_record=False,num_subjects=None):
    h5_path = f"{file_path}/{filename}.h5"
    print(f'Loading: {h5_path}')
    hf = h5py.File(h5_path, "r")
    rows = []
    subj_count = 0
    for subj in hf.keys():
        subj_grp = hf[subj]
        subj_count+=1

        window_count = 0
        for win_id in subj_grp.keys():
            window_count = window_count+1
            win_grp = subj_grp[win_id]
            rec_id     = win_grp.attrs.get('rec_id', '')
            label      = win_grp.attrs.get('label', -1)
            raw_ppg_fs = win_grp.attrs.get("raw_ppg_fs", np.nan)
            ekg_fs     = win_grp.attrs.get("ekg_fs", np.nan)
            ppg_fs     = win_grp.attrs.get("ppg_fs", np.nan)
            abp_fs     = win_grp.attrs.get("abp_fs", np.nan)
            notes      = win_grp.attrs.get("notes", "")
            # pull out everything you need
            raw_ppg     = win_grp["raw_ppg"][:]
            proc_ppg    = win_grp["proc_ppg"][:]
            raw_ekg     = win_grp["raw_ekg"][:]
            raw_abp     = win_grp["raw_abp"][:]
            # build the window row
            row = {
                "subject": subj,
                "window_id": win_id,
                "window_count": window_count,
                "rec_id": rec_id,
                "label": label,
                "raw_ppg_fs": raw_ppg_fs,
                "ppg_fs_out": ppg_fs,
                "ekg_fs_out": ekg_fs,
                "abp_fs_out": abp_fs,
                "raw_len": len(raw_ppg),
                "proc_len": len(proc_ppg),
                "duration_raw_s": len(raw_ppg) / ppg_fs,
                "duration_proc_s": len(proc_ppg) / ppg_fs,
                "notes": notes
            }
            if store_signals:
                row.update({
                    "raw_ppg": raw_ppg,
                    "proc_ppg": proc_ppg,
                    "raw_ekg": raw_ekg,
                    "raw_abp": raw_abp
                })
            rows.append(row)
        if (num_subjects is not None) and (subj_count == num_subjects):
            break

    df = pd.DataFrame(rows)

    if group_by_record and store_signals:
        # define how to aggregate each column
        agg_dict = {
            "subject":     "first",
            "label":       "first",
            "raw_ppg_fs":  "first",
            "ppg_fs_out":  "first",
            "ekg_fs_out":  "first",
            "abp_fs_out":  "first",
            "notes":       lambda x: list(x),
            "raw_len":     lambda x: np.sum(x),
            "proc_len":    lambda x: np.sum(x),
            "duration_raw_s":  lambda x: np.sum(x),
            "duration_proc_s": lambda x: np.sum(x),
            "raw_ppg":  lambda series: np.concatenate(series.values),
            "proc_ppg": lambda series: np.concatenate(series.values),
            "raw_ekg":  lambda series: np.concatenate(series.values),
            "raw_abp":  lambda series: np.concatenate(series.values),
        }
        df = (df.groupby("rec_id", as_index=False).agg(agg_dict))
    return df