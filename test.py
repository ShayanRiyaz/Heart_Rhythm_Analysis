import os
import requests
import datetime
import numpy as np
import wfdb
from scipy.io import savemat


class MimicDatasetCollator:
    def __init__(self, root_folder=None, extraction_length=None):
        """
        Initializes the collator with default or user–specified paths.
        The default root folder should contain subfolders for each subject.
        """
        # If a root folder is provided, use it; otherwise, set a default.
        self.up = {}
        self.up['paths'] = {}
        self.up['paths']['local'] = {}
        self.up['paths']['web'] = {}
        self.root_folder = root_folder
        self.extraction_length = extraction_length

        # Set local root folder (adjust the path as needed)
        self.up['paths']['local']['root_folder'] = root_folder or f'{self.root_folder}/Data/mimiciii_ppg_af_beat_detection/'
        if not os.path.isdir(self.up['paths']['local']['root_folder']):
            print("Warning: Specified folder does not exist. Please adjust 'up.paths.local.root_folder'.")

        # Online root folder for data (PhysioNet)
        self.up['paths']['web']['root_folder'] = 'https://physionet.org/files/mimic3wdb-matched/1.0/'

        # Local folder in which to save downloaded files
        self.up['paths']['local']['temp_folder'] = os.path.join(self.up['paths']['local']['root_folder'], 'downloaded_files')
        if not os.path.isdir(self.up['paths']['local']['temp_folder']):
            print("\n - Making directory in which to store downloaded data")
            os.makedirs(self.up['paths']['local']['temp_folder'])

        # Settings: required duration (in seconds)
        self.up['settings'] = {}
        self.up['settings']['req_durn'] = 1 * 20 * 60

        self.check_WFDB_installation()

    def collate_dataset(self):
        """
        Main entry point: downloads and collates data from the MIMIC-III Waveform Database
        during Atrial Fibrillation (AF) and sinus rhythm.
        """
        print("\n ~~~ Downloading and collating MIMIC III matched waveform database excerpt ~~~")
        self.download_data(self.up)

    def check_WFDB_installation(self):
        """
        Checks whether the WFDB package is installed.
        """
        try:
            import wfdb  # noqa
            print("\n    - Detected that the Waveform Database Software Package (WFDB) for Python is installed.")
        except ImportError:
            raise ImportError("This code requires the WFDB package. Install it from https://github.com/MIT-LCP/wfdb-python.")

    def download_data(self, up):
        """
        Downloads and collates data for subjects that meet the requirements.
        Many steps (record selection, file download, signal extraction, and duration refinement)
        mimic the MATLAB code.
        """
        print("\n - Downloading data")
        stay_list = self.define_stays_for_download()  # record selection info

        data = []  # container for subject data
        inc_subj_no = 0

        # Cycle through each subject based on the provided stay list
        for subj_no in range(len(stay_list['stayID'])):
            # print(stay_list)
            curr_subj = stay_list['stayID'][subj_no]
            print("\n   - Extracting data for subject : {} ' | Record: {}".format(curr_subj,stay_list['file'][subj_no]))

        
            curr_rec = stay_list['file'][subj_no]
            rec_file_info = self.get_rec_file_info(curr_rec, up)
            rec_file_info = self.get_rec_files_in_period(rec_file_info,
                                                         stay_list['onset_time'][subj_no],
                                                         stay_list['offset_time'][subj_no])
            curr_rec_subfolder = curr_rec[:3]

            possible_files = []
            curr_durn = 0
            at_least_one_file_had_signals = False
            t_overall_deb = None
            next_t_deb = None

            # Loop over file segments; break as soon as one file has the required signals.
            for file_no in range(len(rec_file_info['rec_deb'])):
                # (Special case skip; adjust if necessary)
                if curr_subj == 'p092846' and file_no < 3:
                    continue
                if stay_list['af'][subj_no] == 1:
                    print(f'Filename: {curr_rec}: | AF Status: {stay_list['af'][subj_no]}')
                print("    - file {}".format(file_no + 1))
                curr_file = rec_file_info['rec_name'][file_no]
                use_wfdb_toolbox = False
                if use_wfdb_toolbox:
                    vars_list = []
                else:
                    # Construct URL for header file
                    # rec_header_path = os.path.join(up['paths']['web']['root_folder'],
                    #                                curr_rec_subfolder,
                    #                                curr_subj,
                    #                                curr_file + '.hea')
                    rec_header_path = os.path.join(up['paths']['web']['root_folder'],
                                curr_rec_subfolder,
                                curr_subj,
                                curr_file + '.hea')
                    vars_list = self.get_vars_in_rec(rec_header_path)

                ecg_signals = ['II', 'I', 'V', 'III', 'MCL1']
                if (not any(any(sig in v for v in vars_list) for sig in ecg_signals) or
                        not any('PLETH' in v for v in vars_list)):
                    print(f" (didn't have the signals)| Available Variables: {vars_list}")
                    continue  # Skip this file; do not add it to possible_files.
                else: 
                    print(f"(Has Signals) | Available Variables: {vars_list}")
                # File meets the criteria.
                at_least_one_file_had_signals = True
                t_deb = rec_file_info['rec_deb'][file_no]
                if possible_files:
                    # Check start time consistency.
                    t_offset = (next_t_deb - t_deb).total_seconds()
                    if round(1 / t_offset) != rec_file_info['fs']:
                        print(" (didn't have the required start time)")
                        continue
                else:
                    t_overall_deb = t_deb

                possible_files.append(curr_file)
                curr_durn = int((rec_file_info['rec_fin'][file_no] - t_overall_deb).total_seconds())
                if curr_durn >= up['settings']['req_durn']:
                    break  # Stop once we have a file long enough.
                next_t_deb = rec_file_info['rec_fin'][file_no] + datetime.timedelta(seconds=(1 / rec_file_info['fs']))

            if not possible_files:
                if at_least_one_file_had_signals:
                    print("      - not enough data in recording {} (only {:.1f} mins)".format(curr_rec, curr_durn / 60))
                else:
                    print("      - recording {} didn't have the required signals".format(curr_rec))
                continue

            print("     - Downloading relevant data for recording {}:".format(curr_rec))
            inc_subj_no += 1
            os.chdir(up['paths']['local']['temp_folder'])
            no_samps_required = up['settings']['req_durn'] * rec_file_info['fs'] + 1
            no_samps_downloaded = 0

            subj_data = {}
            subj_data['fix'] = {
                'subj_id': curr_subj,
                'rec_id': curr_rec,
                'files': possible_files,
                'af_status': stay_list['af'][subj_no]
            }
            subj_data['ppg'] = {}
            subj_data['ekg'] = {}
            subj_data['imp'] = {}
            subj_data['abp'] = {}

            # Instead of processing all files, choose one file (for example, the first one).
            desired_index = 0
            if len(possible_files) > desired_index:
                curr_file = possible_files[desired_index]
                print("      - Processing file {} directly".format(desired_index + 1))
                filename = curr_file #+ 'm'
                # # Check that the file header exists
                # if not os.path.exists(filename + '.hea'):
                #     print("File {} not found. Skipping subject {}.".format(filename, curr_subj))
                #     continue
                # try:
                #     record = wfdb.rdrecord(filename)
                # except Exception as e:
                #     print("Error reading file {}: {}".format(filename, e))
                #     continue


            # Attempt to download the .hea and .dat files as needed (this part is unchanged)
            hea_url = f"{up['paths']['web']['root_folder']}/{curr_rec_subfolder}/{curr_rec[:7]}/{curr_file}.hea"
            dat_url = f"{up['paths']['web']['root_folder']}/{curr_rec_subfolder}/{curr_rec[:7]}/{curr_file}.dat"

            try:
                if not os.path.exists(f'{filename}.hea'):
                    r = requests.get(hea_url)
                    r.raise_for_status()
                    with open(filename + '.hea', 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print(f"Error downloading header file {filename}.hea from {hea_url}: {e}")
                continue

            try:
                if not os.path.exists(f'{filename}.dat'):
                    r = requests.get(dat_url)
                    r.raise_for_status()
                    with open(filename + '.dat', 'wb') as f:
                        f.write(r.content)
            except Exception as e:
                print(f"Error downloading data file {filename}.dat from {dat_url}: {e}")
                continue

            # Now attempt to read the file
            try:
                record = wfdb.rdrecord( )  # This reads filename.hea and filename.dat
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

            # Now extract the signals
            sig_names = record.sig_name

            # Extract PPG signal
            if 'PLETH' in sig_names:
                idx_ppg = sig_names.index('PLETH')
                subj_data['ppg']['v'] = record.p_signal[:, idx_ppg]
                subj_data['ppg']['fs'] = record.fs
                subj_data['ppg']['method'] = 'Fingertip PPG recorded using bedside monitor'

            # Extract ECG signal: try the listed leads
            ecg_labels = ['II', 'I', 'V', 'III', 'MCL1']
            if curr_subj in ['p00019', 'p00035']:
                ecg_labels = ['V', 'II', 'I', 'III', 'MCL1']
            for label in ecg_labels:
                if label in sig_names:
                    idx_ecg = sig_names.index(label)
                    subj_data['ekg']['v'] = record.p_signal[:, idx_ecg]
                    subj_data['ekg']['fs'] = record.fs
                    subj_data['ekg']['method'] = 'ECG recorded using bedside monitor, lead ' + label
                    subj_data['ekg']['label'] = label
                    break

            # Extract RESP signal if present
            if 'RESP' in sig_names:
                idx_resp = sig_names.index('RESP')
                subj_data['imp']['v'] = record.p_signal[:, idx_resp]
                subj_data['imp']['fs'] = record.fs
                subj_data['imp']['method'] = 'Impedance pneumography respiratory signal recorded at the chest using bedside monitor'

            # Extract ABP signal if present (note the corrected variable name idx_abp)
            if 'ABP' in sig_names:
                idx_abp = sig_names.index('ABP')
                subj_data['abp']['v'] = record.p_signal[:, idx_abp]
                subj_data['abp']['fs'] = record.fs
                subj_data['abp']['method'] = 'Invasive arterial blood pressure recorded using bedside monitor'
            # else:
            #     print("No file available for the given index")
            #     continue

            # (Additional file downloading code below is kept for legacy purposes.)
            idx = rec_file_info['rec_name'].index(curr_file)
            no_samps_in_file = rec_file_info['rec_no_samps'][idx]
            remaining_samps_required = no_samps_required - no_samps_downloaded
            no_samps_to_download = min(no_samps_in_file, remaining_samps_required)

            filename = curr_file #+ 'm'
            if not os.path.exists(filename + '.hea'):
                print("File {} not found. Skipping file.".format(filename))
                continue
            if not os.path.exists(filename + '.mat'):
                specify_no_samps = False
                if specify_no_samps:
                    # Optionally download a specified number of samples.
                    pass
                else:
                    try:
                        # file_path = os.path.join(os.getcwd(),'mimic3wdb', 'matched', curr_rec_subfolder, curr_rec[:7])
                        # if not os.path.exists(file_path):
                            # os.mkdir(file_path)
                        # record = wfdb.rdrecord(os.path.join('mimic3wdb', 'matched', curr_rec_subfolder, curr_rec[:7], curr_file))
                        record = wfdb.rdrecord(os.path.join(curr_file))
                        
                        savemat(filename + '.mat', {'signal': record.p_signal, 'fs': record.fs, 'sig_names': record.sig_name})
                    except Exception as e:
                        print("Error downloading file {}: {}".format(curr_file, e))
                        continue
            no_samps_downloaded += no_samps_to_download

            # Double-check reading the downloaded file
            try:
                record = wfdb.rdrecord(filename)
            except Exception as e:
                print("Error reading file {}: {}".format(filename, e))
                continue
            sig_names = record.sig_name

            # In case additional signal concatenation is needed (not applicable here since we process one file):
            if 'v' not in subj_data['ppg'] and 'PLETH' in sig_names:
                idx_ppg = sig_names.index('PLETH')
                subj_data['ppg']['v'] = record.p_signal[:, idx_ppg]
                subj_data['ppg']['fs'] = record.fs

            if 'v' in subj_data.get('ppg', {}):
                print("\n Exported data lasting {} mins".format(
                    round(len(subj_data['ppg']['v']) / (60 * subj_data['ppg']['fs']))))
            else:
                print("\n No PPG signal data extracted for subject {}.".format(curr_subj))
                continue

            # Reduce duration if necessary.
            file_duration = len(subj_data['ppg']['v']) / subj_data['ppg']['fs']
            if self.extraction_length is None:
                durn = file_duration
            else:
                fixed_durn = self.extraction_length * 60
                durn = fixed_durn if file_duration > fixed_durn else file_duration

            win_step = 60  # window step in seconds
            if file_duration > durn:
                S = {'v': subj_data['ppg']['v'], 'fs': subj_data['ppg']['fs']}
                S = self.identify_periods_of_no_signal(S)
                S2 = {'v': subj_data['ekg']['v'], 'fs': subj_data['ekg']['fs']}
                S2 = self.identify_periods_of_no_signal(S2)
                fs = subj_data['ppg']['fs']
                win_starts_ppg = np.arange(0, len(S['v']) - int(durn * fs), int(win_step * fs))
                win_ends_ppg = win_starts_ppg + int(durn * fs)
                n_no_signal = np.empty(len(win_starts_ppg))
                n_no_signal.fill(np.nan)
                for win_no, (start, end) in enumerate(zip(win_starts_ppg, win_ends_ppg)):
                    win_els = np.arange(start, end)
                    n_no_signal[win_no] = (np.sum(S.get('no_signal', np.zeros_like(S['v']))[win_els]) +
                                           np.sum(S2.get('no_signal', np.zeros_like(S2['v']))[win_els]))
                    if np.isnan(S['v'][win_els]).any() or np.isnan(S2['v'][win_els]).any():
                        n_no_signal[win_no] += 2 * len(win_els)
                most_complete_win = np.argmin(n_no_signal)
                rel_win_start = win_starts_ppg[most_complete_win]
                rel_win_end = win_ends_ppg[most_complete_win]
                req_samps = np.arange(rel_win_start, rel_win_end)
                for sig in ['ekg', 'ppg', 'imp', 'abp']:
                    if sig in subj_data and subj_data[sig]:
                        subj_data[sig]['v'] = subj_data[sig]['v'][req_samps]
                print("\n Refined to {} mins".format(
                    round(len(subj_data['ppg']['v']) / (60 * subj_data['ppg']['fs']))))

            data.append(subj_data)

        # Save the collated data into two files (AF and non-AF)
        source = {
            'date_of_conversion': datetime.datetime.now().isoformat(),
            'matlab_conversion_script': __file__,
            'raw_data_path': up['paths']['local']['temp_folder']
        }
        license_info = {}
        license_info['readme'] = "Use the following command to view the license: print(license_info['details'])"
        license_info['details'] = (
            "This dataset is licensed under the Open Data Commons Open Database License v1.0 (ODbL 1.0 license). "
            "Further details are available at: https://opendatacommons.org/licenses/odbl/summary/\n\n"
            "This dataset is derived from the MIMIC III Waveform Database:\n"
            "Moody, B., Moody, G., Villarroel, M., Clifford, G. D., & Silva, I. (2020). MIMIC-III Waveform Database (version 1.0). PhysioNet. https://doi.org/10.13026/c2607m.\n\n"
            "The MIMIC III Waveform Database is licensed under the ODbL 1.0 license.\n\n"
            "The MIMIC-III database is described in the following publication:\n"
            "Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., & Mark, R. G. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035. https://doi.org/10.1038/sdata.2016.35\n\n"
            "It is available on PhysioNet, https://physionet.org/ :\n"
            "Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220.\n\n"
            "The following annotations of AF and non-AF were used to create the dataset:\n\n"
            "Bashar, Syed Khairul (2020): Atrial Fibrillation annotations of electrocardiogram from MIMIC III matched subset. figshare. Dataset. https://doi.org/10.6084/m9.figshare.12149091.v1\n\n"
            "Bashar, S.K., Ding, E., Walkey, A.J., McManus, D.D. and Chon, K.H., 2019. Noise Detection in Electrocardiogram Signals for Intensive Care Unit Patients. IEEE Access, 7, pp.88357-88368. https://doi.org/10.1109/ACCESS.2019.2926199\n\n"
            "This annotation information is reproduced under the terms of the CC BY 4.0 licence: https://creativecommons.org/licenses/by/4.0/"
        )

        af_data = [d for d in data if d['fix']['af_status'] == 1]
        non_af_data = [d for d in data if d['fix']['af_status'] == 0]
        print("\n   - Saving data: ", end="")
        print("AF ({} subjs)".format(len(af_data)))
        savemat(os.path.join(up['paths']['local']['root_folder'], 'mimic_af_data.mat'),
                {'data': af_data, 'source': source, 'license': license_info})
        print(", non-AF ({} subjs)".format(len(non_af_data)))
        savemat(os.path.join(up['paths']['local']['root_folder'], 'mimic_non_af_data.mat'),
                {'data': non_af_data, 'source': source, 'license': license_info})

        print("\n\n - NB: this dataset also contains additional variables which have not been imported")
        print("\n\n ~~~ DONE ~~~")

    def get_vars_in_rec(self, rec_header_path):
        """
        Downloads and parses a header file to extract signal variable names.
        """
        try:
            response = requests.get(rec_header_path)
            response.raise_for_status()
        except Exception as e:
            print("Error downloading header file from {}: {}".format(rec_header_path, e))
            return []
        sig_data = response.text.splitlines()
        vars_list = []
        while sig_data and sig_data[0].endswith(' '):
            sig_data[0] = sig_data[0].rstrip()
        for line in sig_data[1:]:#[1:-1]:
            parts = line.split()
            if len(parts) >= 2:
                vars_list.append(parts[-1])
        return vars_list

    def get_rec_file_info(self, curr_rec, up):
        """
        Downloads and parses the header for a particular recording.
        """
        curr_rec_subfolder = curr_rec[:3]
        curr_rec_hea_path = os.path.join(up['paths']['web']['root_folder'],
                                         curr_rec_subfolder,
                                         curr_rec[:7],
                                         curr_rec + '.hea')
        try:
            response = requests.get(curr_rec_hea_path)
            response.raise_for_status()
        except Exception as e:
            print("Error downloading record header from {}: {}".format(curr_rec_hea_path, e))
            return {}
        rec_data = response.text.splitlines()
        header_parts = rec_data[0].split()
        rec_file_info = {}
        try:
            rec_file_info['deb'] = datetime.datetime.strptime(header_parts[4] + ' ' + header_parts[5],
                                                                "%H:%M:%S.%f %d/%m/%Y")
        except ValueError:
            rec_file_info['deb'] = datetime.datetime.strptime(header_parts[4] + ' ' + header_parts[5],
                                                                "%H:%M:%S %d/%m/%Y")
        rec_file_info['no_sigs'] = int(header_parts[1])
        rec_file_info['fs'] = float(header_parts[2])
        rec_file_info['no_samps'] = int(header_parts[3])
        rec_file_info['fin'] = rec_file_info['deb'] + datetime.timedelta(seconds=rec_file_info['no_samps'] / rec_file_info['fs'])

        no_samps_passed = 0
        rec_file_info['rec_name'] = []
        rec_file_info['rec_deb'] = []
        rec_file_info['rec_no_samps'] = []
        rec_file_info['rec_fin'] = []
        rec_file_info['prev_blank'] = []
        rec_file_info['next_blank'] = []
        prev_seg_blank = False
        line_no = 1
        while line_no < len(rec_data):
            line = rec_data[line_no]
            line_no += 1
            if not line or line.startswith('#') or 'layout' in line:
                continue
            if line.startswith('~'):
                no_samps_passed += int(line[1:])
                prev_seg_blank = True
                if rec_file_info['rec_name']:
                    rec_file_info['next_blank'][-1] = 1
                continue
            parts = line.split()
            rec_file_info['rec_name'].append(parts[0])
            rec_file_info['rec_no_samps'].append(int(parts[1]))
            rec_file_info['rec_deb'].append(rec_file_info['deb'] + datetime.timedelta(seconds=no_samps_passed / rec_file_info['fs']))
            no_samps_passed += int(parts[1])
            rec_file_info['rec_fin'].append(rec_file_info['deb'] + datetime.timedelta(seconds=no_samps_passed / rec_file_info['fs']))
            rec_file_info['prev_blank'].append(1 if prev_seg_blank else 0)
            rec_file_info['next_blank'].append(0)
            prev_seg_blank = False
        return rec_file_info

    def get_rec_files_in_period(self, rec_file_info, onset_time, offset_time):
        """
        Identifies file segments that are completely within the desired period.
        """
        period_deb = rec_file_info['deb']
        if onset_time:
            t_onset = datetime.datetime.strptime(onset_time, "%H:%M:%S")
            period_deb = period_deb.replace(hour=t_onset.hour, minute=t_onset.minute, second=t_onset.second)
        if offset_time:
            parts = offset_time.split(':')
            hours, minutes, seconds = map(int, parts)
            delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
            period_fin = period_deb + delta
        else:
            period_fin = datetime.datetime.max

        in_period = []
        for deb, fin in zip(rec_file_info['rec_deb'], rec_file_info['rec_fin']):
            in_period.append(deb >= period_deb and fin <= period_fin)
        rec_file_info['in_period'] = in_period
        return rec_file_info

    def identify_periods_of_no_signal(self, S):
        """
        Identifies periods when there is no (or flat) signal using a sliding-window approach.
        """
        durn_flat_line = 0.2  # seconds
        fs = S['fs']
        v = S['v']
        not_nan_log = ~np.isnan(v)
        diff_v = np.diff(v, prepend=v[0])
        not_flat_line_log = diff_v != 0

        env_t = 1  # seconds
        env_samps = int(round(env_t * fs))
        upper_env = np.array([np.max(v[max(0, i - env_samps):min(len(v), i + env_samps)])
                              for i in range(len(v))])
        lower_env = np.array([np.min(v[max(0, i - env_samps):min(len(v), i + env_samps)])
                              for i in range(len(v))])
        on_env = (v == upper_env) | (v == lower_env)
        mask = ~not_flat_line_log & (~on_env)
        not_flat_line_log[mask] = True

        window_size = int(round(fs * durn_flat_line))
        no_signal = np.convolve(~(not_flat_line_log & not_nan_log), np.ones(window_size, dtype=int), mode='same') > window_size
        S['no_signal'] = no_signal.astype(int)
        return S

    def define_stays_for_download(self):
        """
        Defines which MIMIC III records to download based on annotations.
        """
        records = {}
        records['af'] = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0,
                                  0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1])
        records['subj_id'] = np.array([608, 776, 946, 4490, 4829, 75796, 9526, 10391, 13072,
                                       13136, 14079, 15852, 16684, 17344, 19608, 22954, 23824,
                                       25117, 26377, 26964, 29512, 43613, 50089, 50384, 55204,
                                       58932, 62160, 63039, 63628, 68956, 69339, 75371, 77729,
                                       87275, 79998, 81349, 85866, 87675, 89565, 89964, 92289,
                                       92846, 94847, 97547, 99674])
        records['file'] = ['p000608-2167-03-09-11-54', 'p000776-2184-04-30-15-16',
                           'p000946-2120-05-14-08-08', 'p004490-2151-01-07-12-36',
                           'p004829-2103-08-30-21-52', 'p075796-2198-07-25-23-40',
                           'p009526-2113-11-17-02-12', 'p010391-2183-12-25-10-15',
                           'p013072-2194-01-22-16-13', 'p013136-2133-11-09-16-58',
                           'p014079-2182-09-24-13-41', 'p015852-2148-05-03-18-39',
                           'p016684-2188-01-29-00-06', 'p017344-2169-07-17-17-32',
                           'p019608-2125-02-05-04-57', 'p022954-2136-02-29-17-52',
                           'p023824-2182-11-27-14-22', 'p025117-2202-03-15-20-28',
                           'p026377-2111-11-17-16-46', 'p026964-2147-01-11-18-03',
                           'p029512-2188-02-27-18-10', 'p043613-2185-01-18-23-52',
                           'p050089-2157-08-23-16-37', 'p050384-2195-01-30-02-21',
                           'p055204-2132-06-30-09-34', 'p058932-2120-10-13-23-15',
                           'p062160-2153-10-03-14-49', 'p063039-2157-03-29-13-35',
                           'p063628-2176-07-02-20-38', 'p068956-2107-04-21-16-05',
                           'p069339-2133-12-09-21-14', 'p075371-2119-08-22-00-53',
                           'p077729-2120-08-31-01-03', 'p087275-2108-08-29-12-53',
                           'p079998-2101-10-21-21-31', 'p081349-2120-02-11-06-35',
                           'p085866-2178-03-20-17-11', 'p087675-2104-12-05-03-53',
                           'p089565-2174-05-12-00-07', 'p089964-2154-05-21-14-53',
                           'p092289-2183-03-17-23-12', 'p092846-2129-12-21-13-12',
                           'p094847-2112-02-12-19-56', 'p097547-2125-10-21-23-43',
                           'p099674-2105-06-13-00-07']
        records['onset_time'] = ['', '', '00:00:00', '00:00:00', '', '00:00:00',
                                 '00:00:28', '', '', '', '00:00:00', '', '', '', '',
                                 '00:00:00', '', '00:01:30', '00:03:41', '', '', '00:00:49',
                                 '00:06:29', '', '', '', '', '00:00:00', '', '00:00:03',
                                 '00:02:47', '00:28:07', '00:00:27', '00:00:17', '00:01:11',
                                 '00:00:00', '00:01:30', '', '00:00:51', '', '', '00:00:00',
                                 '00:01:14', '', '00:00:07']
        records['offset_time'] = ['', '', '06:13:12', '08:54:53', '', '42:56:15',
                                  '05:29:00', '', '', '', '03:57:49', '', '', '', '',
                                  '22:54:21', '', '12:38:58', '29:59:44', '', '', '30:16:16',
                                  '44:01:00', '', '', '', '', '30:20:34', '', '18:50:22',
                                  '08:39:47', '14:06:19', '15:20:01', '12:14:49', '20:16:36',
                                  '05:02:36', '23:38:35', '', '13:00:29', '', '', '12:17:40',
                                  '22:52:35', '', '11:53:00']
        records['stayID'] = [f[:7] for f in records['file']]
        print(records['stayID'])
        return records


if __name__ == '__main__':
    # Instantiate the collator and run the collate_dataset() method.
    collator = MimicDatasetCollator(root_folder=os.getcwd(), extraction_length=None)
    collator.collate_dataset()