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
        self.up = {}
        self.up['paths'] = {}
        self.up['paths']['local'] = {}
        self.up['paths']['web'] = {}
        self.root_folder = root_folder
        self.extraction_length = extraction_length

        # Set local root folder (adjust the path as needed)
        self.up['paths']['local']['root_folder'] = root_folder or f'{os.getcwd()}/Data/mimiciii_ppg_af_beat_detection/'
        if not os.path.isdir(self.up['paths']['local']['root_folder']):
            print("Warning: Specified folder does not exist. Please adjust 'up.paths.local.root_folder'.")

        # Online root folder for data (PhysioNet)
        self.up['paths']['web']['root_folder'] = 'https://physionet.org/files/mimic3wdb-matched/1.0/'

        # Local folder in which to save downloaded files
        self.up['paths']['local']['temp_folder'] = os.path.join(self.up['paths']['local']['root_folder'], 'downloaded_files','all_files')
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

        # Cycle through each subject based on the provided stay list
        for subj_no in range(len(stay_list['stayID'])):
            curr_subj = stay_list['stayID'][subj_no]
            print("\n   - Extracting data for subject : {} | Record: {}".format(curr_subj, stay_list['file'][subj_no]))

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

            # Loop over file segments (for selection) as in MATLAB.
            for file_no in range(len(rec_file_info['rec_deb'])):
                # (MATLAB skip condition: if subj_no==35 && file_no==1 || subj_no==22 && file_no<5)
                # In Python (0-indexed): if (subj_no==34 and file_no==0) or (subj_no==21 and file_no<4), then skip.
                if (subj_no == 34 and file_no == 0) or (subj_no == 21 and file_no < 4):
                    continue

                print("    - file {}".format(file_no + 1))
                curr_file = rec_file_info['rec_name'][file_no]

                # Download header to check available variables.
                rec_header_path = os.path.join(up['paths']['web']['root_folder'],
                                               curr_rec_subfolder,
                                               curr_subj,
                                               curr_file + '.hea')
                vars_list = self.get_vars_in_rec(rec_header_path)

                ecg_signals = ['II', 'I', 'V', 'III', 'MCL1']
                if (not any(any(sig in v for v in vars_list) for sig in ecg_signals) or
                        not any('PLETH' in v for v in vars_list)):
                    print(f"     (didn't have the signals) | Available Variables: {vars_list}")
                    continue  # Skip file if required signals are not available
                else:
                    print(f"     (Has Signals) | Available Variables: {vars_list}")

                at_least_one_file_had_signals = True
                t_deb = rec_file_info['rec_deb'][file_no]
                if possible_files:
                    t_offset = (next_t_deb - t_deb).total_seconds()
                    if round(1 / t_offset) != rec_file_info['fs']:
                        print("     (didn't have the required start time)")
                        continue
                else:
                    t_overall_deb = t_deb

                possible_files.append(curr_file)
                curr_durn = int((rec_file_info['rec_fin'][file_no] - t_overall_deb).total_seconds())
                if curr_durn >= up['settings']['req_durn']:
                    break  # Stop once we have enough duration.
                next_t_deb = rec_file_info['rec_fin'][file_no] + datetime.timedelta(seconds=(1 / rec_file_info['fs']))

            if not possible_files:
                if at_least_one_file_had_signals:
                    print("      - not enough data in recording {} (only {:.1f} mins)".format(curr_rec, curr_durn / 60))
                else:
                    print("      - recording {} didn't have the required signals".format(curr_rec))
                continue

            print("     - Downloading relevant data for recording {}:".format(curr_rec))
            temp_folder = up['paths']['local']['temp_folder']
            no_samps_required = up['settings']['req_durn'] * rec_file_info['fs'] + 1
            no_samps_downloaded = 0

            # Initialize subject data with fixed fields
            subj_data = {
                'fix': {
                    'subj_id': curr_subj,
                    'rec_id': curr_rec,
                    'files': possible_files,
                    'af_status': stay_list['af'][subj_no]
                },
                'ppg': {},
                'ekg': {},
                'imp': {},
                'abp': {}
            }

            # Process each possible file.
            processed_file_index = 0
            for file_no, curr_file in enumerate(possible_files):
                # Apply MATLAB’s file-skip condition (already applied during file selection?
                # If needed in processing, we keep the same logic)
                if (subj_no == 34 and file_no == 0) or (subj_no == 21 and file_no < 4):
                    continue

                print("      - Processing file {} for subject {}".format(processed_file_index + 1, curr_subj))
                filename_base = os.path.join(temp_folder, curr_file)

                # Download header and data files (if not present)
                hea_url = os.path.join(up['paths']['web']['root_folder'],
                                       curr_rec_subfolder,
                                       curr_subj,
                                       curr_file + '.hea')
                dat_url = os.path.join(up['paths']['web']['root_folder'],
                                       curr_rec_subfolder,
                                       curr_subj,
                                       curr_file + '.dat')

                if not os.path.exists(filename_base + '.hea'):
                    try:
                        r = requests.get(hea_url)
                        r.raise_for_status()
                        with open(filename_base + '.hea', 'wb') as f:
                            f.write(r.content)
                    except Exception as e:
                        print(f"Error downloading header file {filename_base}.hea from {hea_url}: {e}")
                        continue

                if not os.path.exists(filename_base + '.dat'):
                    try:
                        r = requests.get(dat_url)
                        r.raise_for_status()
                        with open(filename_base + '.dat', 'wb') as f:
                            f.write(r.content)
                    except Exception as e:
                        print(f"Error downloading data file {filename_base}.dat from {dat_url}: {e}")
                        continue

                # Read the record using WFDB.
                try:
                    record = wfdb.rdrecord(filename_base)
                except Exception as e:
                    print(f"Error reading file {filename_base}: {e}")
                    continue

                sig_names = record.sig_name

                # Determine number of samples for this file.
                try:
                    idx_file = rec_file_info['rec_name'].index(curr_file)
                except ValueError:
                    print("Couldn't match current file {} in record info.".format(curr_file))
                    continue
                no_samps_in_file = rec_file_info['rec_no_samps'][idx_file]
                remaining_samps_required = no_samps_required - no_samps_downloaded
                no_samps_to_download = min(no_samps_in_file, remaining_samps_required)
                no_samps_downloaded += no_samps_to_download

                # Now extract and assign signals:
                if (processed_file_index == 0 or 
                    (subj_no == 34 and processed_file_index == 1) or 
                    (subj_no == 21 and processed_file_index == 4)):
                    # (MATLAB: if file_no == 1 or (subj_no == 35 && file_no == 2) or (subj_no == 22 && file_no == 5))
                    # Initialize/assign signals from the first (or designated) file.
                    if 'PLETH' in sig_names:
                        idx_ppg = sig_names.index('PLETH')
                        subj_data['ppg']['v'] = record.p_signal[:, idx_ppg]
                        subj_data['ppg']['fs'] = record.fs
                        subj_data['ppg']['method'] = 'Fingertip PPG recorded using bedside monitor'
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
                    if 'RESP' in sig_names:
                        idx_resp = sig_names.index('RESP')
                        subj_data['imp']['v'] = record.p_signal[:, idx_resp]
                        subj_data['imp']['fs'] = record.fs
                        subj_data['imp']['method'] = 'Impedance pneumography respiratory signal recorded at the chest using bedside monitor'
                    if 'ABP' in sig_names:
                        idx_abp = sig_names.index('ABP')
                        subj_data['abp']['v'] = record.p_signal[:, idx_abp]
                        subj_data['abp']['fs'] = record.fs
                        subj_data['abp']['method'] = 'Invasive arterial blood pressure recorded using bedside monitor'
                else:
                    # Additional files – concatenate signals.
                    if 'PLETH' in sig_names and 'v' in subj_data['ppg']:
                        idx_ppg = sig_names.index('PLETH')
                        subj_data['ppg']['v'] = np.concatenate((subj_data['ppg']['v'], record.p_signal[:, idx_ppg]), axis=0)
                    if 'ekg' in subj_data and 'v' in subj_data['ekg']:
                        curr_ecg_label = subj_data['ekg'].get('label', None)
                        if curr_ecg_label and curr_ecg_label in sig_names:
                            idx_ecg = sig_names.index(curr_ecg_label)
                            subj_data['ekg']['v'] = np.concatenate((subj_data['ekg']['v'], record.p_signal[:, idx_ecg]), axis=0)
                        else:
                            print("Error: Could not find required ECG signal in additional file")
                    if 'RESP' in sig_names and 'v' in subj_data['imp']:
                        idx_resp = sig_names.index('RESP')
                        subj_data['imp']['v'] = np.concatenate((subj_data['imp']['v'], record.p_signal[:, idx_resp]), axis=0)
                    if 'ABP' in sig_names and 'v' in subj_data['abp']:
                        idx_abp = sig_names.index('ABP')
                        subj_data['abp']['v'] = np.concatenate((subj_data['abp']['v'], record.p_signal[:, idx_abp]), axis=0)

                processed_file_index += 1

            # (Optional) Duration refinement using a no-signal identification function.
            file_duration = len(subj_data['ppg'].get('v', [])) / record.fs if 'v' in subj_data['ppg'] else 0
            if self.extraction_length is not None and file_duration > self.extraction_length * 60:
                # Refine the duration of the signals:
                S = {'v': subj_data['ppg']['v'], 'fs': subj_data['ppg']['fs']}
                S = self.identify_periods_of_no_signal(S)
                S2 = {'v': subj_data['ekg']['v'], 'fs': subj_data['ekg']['fs']}
                S2 = self.identify_periods_of_no_signal(S2)
                fs = subj_data['ppg']['fs']
                win_step = 60  # window step in seconds
                win_starts = np.arange(0, len(S['v']) - int(self.extraction_length * fs), int(win_step * fs))
                win_ends = win_starts + int(self.extraction_length * fs)
                n_no_signal = np.empty(len(win_starts))
                n_no_signal.fill(np.nan)
                for i, (start, end) in enumerate(zip(win_starts, win_ends)):
                    win_indices = np.arange(start, end)
                    n_no_signal[i] = (np.sum(S.get('no_signal', np.zeros_like(S['v']))[win_indices]) +
                                      np.sum(S2.get('no_signal', np.zeros_like(S2['v']))[win_indices]))
                    if np.isnan(S['v'][win_indices]).any() or np.isnan(S2['v'][win_indices]).any():
                        n_no_signal[i] += 2 * len(win_indices)
                most_complete_win = np.argmin(n_no_signal)
                req_indices = np.arange(win_starts[most_complete_win], win_ends[most_complete_win])
                for key in ['ppg', 'ekg', 'imp', 'abp']:
                    if 'v' in subj_data.get(key, {}):
                        subj_data[key]['v'] = subj_data[key]['v'][req_indices]

                refined_duration = len(subj_data['ppg']['v']) / subj_data['ppg']['fs']
                print("     - Refined to {:.1f} mins".format(refined_duration))
            else:
                print("     - Exported data lasting {:.1f} mins".format(file_duration / 60))
            data.append(subj_data)

        # Separate AF and non-AF data.
        af_data = [d for d in data if d['fix']['af_status'] == 1]
        non_af_data = [d for d in data if d['fix']['af_status'] == 0]
        if self.extraction_length is None:
            file_size = 'Full'
        else:
            file_size = str(self.extraction_length)
        FILE_SAVE_PATH = os.path.join(up['paths']['local']['root_folder'],'downloaded_files',f'length_{file_size}')
        if not os.path.exists(FILE_SAVE_PATH):
            os.mkdir(FILE_SAVE_PATH)

        print("\n   - Saving data: AF ({} subjs)".format(len(af_data)))
        savemat(os.path.join(FILE_SAVE_PATH, 'mimic_af_data.mat'),
                {'data': af_data, 'source': {'date_of_conversion': datetime.datetime.now().isoformat()}, 'license': {}})
        print("   - Saving data: non-AF ({} subjs)".format(len(non_af_data)))
        savemat(os.path.join(FILE_SAVE_PATH, 'mimic_non_af_data.mat'),
                {'data': non_af_data, 'source': {'date_of_conversion': datetime.datetime.now().isoformat()}, 'license': {}})

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
        # Remove trailing spaces
        sig_data = [line.rstrip() for line in sig_data]
        vars_list = []
        for line in sig_data[1:]:
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
    collator = MimicDatasetCollator(root_folder=os.getcwd(), extraction_length=5)
    collator.collate_dataset()