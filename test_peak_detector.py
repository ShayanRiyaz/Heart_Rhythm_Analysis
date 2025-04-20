import numpy as np
from mmpdv2_beat_detector_old import MMPDV2BeatDetector, mmpdv2_beat_detector

# ---------------------------------------------
#  Config
# ---------------------------------------------
fs = 100               # sampling rate, Hz
duration = 30          # seconds
t = np.arange(0, duration, 1/fs)

# ---------------------------------------------
#  Fake PPG (for real work, load/record your own)
#  Here: a 1.2‑Hz sine (≈72 bpm) + noise
# ---------------------------------------------
heart_rate_hz = 1.2
sig = 0.7 * np.sin(2 * np.pi * heart_rate_hz * t)          # pulsatile part
sig += 0.3 * np.sin(2 * np.pi * 2 * heart_rate_hz * t)     # harmonic
sig += 0.05 * np.random.randn(t.size)                      # broadband noise
# ---------------------------------------------
#  Run the detector – two equivalent APIs
# ---------------------------------------------
detector = MMPDV2BeatDetector()        # OO style
peaks, onsets = detector.detect(sig, fs)

# functional wrapper (identical result):
# peaks, onsets = mmpdv2_beat_detector(sig, fs)

print(f"Detected {len(peaks)} beats in {duration} s (~{60*len(peaks)/duration:.1f} bpm)")
print(f"First five peak indices: {peaks[:5]}")
print(f"First five onset indices:{onsets[:5]}")

import matplotlib.pyplot as plt

peak_times = peaks / fs
plt.figure()
plt.plot(t,sig)
plt.plot(peak_times, sig[peaks], 'ro', label='Detected peaks')
plt.show()