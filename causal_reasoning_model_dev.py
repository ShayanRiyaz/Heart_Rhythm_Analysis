import numpy as np
import pandas as pd
from scipy.signal import welch
from dowhy import CausalModel
import time



from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PPGWindowModel import PPGWindow
from utils import *


# --- Utility functions ---
def band_power(f, Pxx, low, high):
    """
    Compute band power between 'low' and 'high' frequencies from PSD.
    """
    mask = (f >= low) & (f <= high)
    return np.trapz(Pxx[mask], f[mask])


def sample_entropy(rr, m=2, r=0.2):
    """
    Compute sample entropy of RR intervals.
    """
    N = len(rr)
    rr = np.array(rr)
    r *= np.std(rr, ddof=1)
    def _phi(m):
        x = np.array([rr[i:i+m] for i in range(N-m+1)])
        count = 0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            count += np.sum(dist <= r) - 1
        return count
    return -np.log(_phi(m+1) / _phi(m))


def fractal_dimension(sig):
    """
    Estimate fractal dimension (Higuchi's method).
    Placeholder – implement as needed.
    """
    # ...
    return np.nan


def detect_peaks(ppg_signal, fs=100):
    """
    Placeholder peak detection. Replace with your NN or algorithm.
    Returns indices of detected peaks.
    """
    # Example: find local maxima above threshold
    # ...
    return []

# --- Feature extraction ---
def extract_features(peaks, ppg_signal, fs=100):
    # RR intervals (ms)
    rr = np.diff(peaks) / fs * 1000
    hr = 60000 / rr

    # Time-domain HRV
    feats = {
        'HR_mean': np.mean(hr),
        'SDNN': np.std(rr, ddof=1),
        'RMSSD': np.sqrt(np.mean(np.diff(rr)**2)),
        'pNN50': np.mean(np.abs(np.diff(rr)) > 50) * 100
    }

    # Frequency-domain HRV
    fs_rr = 1000.0 / np.median(rr)
    f, Pxx = welch(rr, fs=fs_rr, nperseg=len(rr)//2)
    feats.update({
        'LF': band_power(f, Pxx, 0.04, 0.15),
        'HF': band_power(f, Pxx, 0.15, 0.4),
        'LF_HF': band_power(f, Pxx, 0.04, 0.15) / band_power(f, Pxx, 0.15, 0.4)
    })

    # Non-linear / chaos
    feats['SampleEntropy'] = sample_entropy(rr)
    feats['FractalDim'] = fractal_dimension(ppg_signal)

    # Windowed statistics (example over whole signal)
    feats['RR_std'] = np.std(rr)
    feats['HR_skew'] = pd.Series(hr).skew()

    # Add morphology features or others as needed
    # ...

    return feats

# --- Offline training ---
def train_causal_model(X, y, treatment='SampleEntropy'):
    df = pd.DataFrame(X)
    df['outcome'] = y

    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome='outcome',
        graph="""digraph {
            age;
            sex;
            {} -> outcome;
            age -> outcome;
            sex -> outcome;
        }""".format(treatment)
    )
    estimand = model.identify_effect()
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.propensity_score_matching"
    )
    return model, estimate

# --- Real-time prediction loop ---
def real_time_afib_risk(new_signal,new_peaks,estimate, fs=100, window_size=15, step=5):
    """
    Stream PPG, extract features every `step` seconds on a sliding window of `window_size`.
    Print AFib risk probability.
    """
    buffer_peaks = []
    buffer_signal = []  # raw PPG values
    window_samples = window_size * fs

    while True:
        # Acquire new data (implement PPG data fetch)
        # new_signal = acquire_ppg_frame()  # user-defined
        # new_peaks = detect_peaks(new_signal, fs)

        buffer_signal.extend(new_signal)
        buffer_peaks.extend(new_peaks)

        # Keep only the last `window_size` seconds
        if len(buffer_signal) > window_samples:
            buffer_signal = buffer_signal[-window_samples:]
            # adjust peak indices relative to new buffer
            buffer_peaks = [p for p in buffer_peaks if p >= len(buffer_signal) * -1]

        # Extract features and estimate risk
        feats = extract_features(buffer_peaks, buffer_signal, fs)
        risk = estimate.do(feats)
        print(f"[{time.strftime('%H:%M:%S')}] AFib risk: {risk:.3f}")

        # time.sleep(step)



# --- Main script usage ---
if __name__ == "__main__":

    FS = 20.83
    WIN_LEN = int(round(30*FS))
    BATCH = 32
    
    # Example: train model
    FOLDER_PATH = 'length_full'
    TRAIN_PATH = f"downloaded_files/{FOLDER_PATH}/train_ds.h5"
    best_config = find_sliding_window(WIN_LEN,target_windows = 5, overlap=25)
    scale_type = "norm"
    train_ds = PPGWindow(TRAIN_PATH,best_config=best_config,scale_type=scale_type)
    train_loader = DataLoader(train_ds, batch_size=BATCH,shuffle=True, num_workers=0, pin_memory=False)
    for xb, yb,original_signal,label in train_loader:
        for curr_xb,curr_yb,curr_original_signal,curr_label in zip(xb,yb,original_signal,label):
            X, y = [], []
            X.append(extract_features(curr_yb, curr_xb, 20.83))
            y.append(curr_label)

    model, estimate = train_causal_model(X, y)
    FOLDER_PATH = 'length_full'
    TEST_PATH = f"downloaded_files/{FOLDER_PATH}/test_ds.h5"
    test_ds = PPGWindow(TEST_PATH,best_config=best_config,scale_type=scale_type)
    test_loader = DataLoader(test_ds, batch_size=BATCH,shuffle=True, num_workers=0, pin_memory=False)
    for xb, yb,original_signal,label in test_loader:
        for curr_xb,curr_yb,curr_original_signal,curr_label in zip(xb,yb,original_signal,label):
            X, y = [], []
            X.append(extract_features(curr_yb, curr_xb, 20.83))
            y.append(curr_label)

    real_time_afib_risk(X,y,estimate)
     
    pass
