
# Heart Rhythm Analysis
By: [Shayan Riyaz](https://shayanriyaz.github.io)


## Motivation:
The purpose of this project is to explore Heart Rhythms using:
1) Signal Processing Techniques
2) Machine/Deep Learning
3) Causal Analysis/Inference
## Dataset
Currently I am using small AFib/NSR Dataset (~35 subjects) from the MIMIC III via physionet. I converted **[Peter Charlton's](https://github.com/peterhcharlton/ppg-beats)** [collate_mimic_perform_af_dataset](https://github.com/peterhcharlton/ppg-beats/blob/main/source/collate_mimic_perform_af_dataset.m) script for extracting the data from Matlab to python. (Using a max of LLMs and my own intuiton). Next I used the [MSPTDfast](https://iopscience.iop.org/article/10.1088/1361-6579/adb89e) peak detection [algorithm](https://github.com/peterhcharlton/ppg-beats/blob/main/source/msptdfastv2_beat_detector.m) (again by Peter Charlton & again converted using LLM and my own understanding of the algorithm.). 

**Datasets Prepared:**
- Training and Validation
    - MIMIC III AFib and Sinus Rhythm Subset (~35 subejects)
    - MIMIC IV AFib and Sinus Rhythm Subject (~50 Subjects)
  
- Testing
  - CapnoBase 


## Model Development
- Currently i created a dummy Conv1D net for peak detection for PPG Signals. (not sure if its necessary but a good learnign exercise).
### Result
![Current Status](assets/image.png)

Goals:
- Noisy Frame Detection using Deep Learning + Simple Statistical Measurments 
- Peak Detection using Semi-supervised (pre-labeled using MSPTDfast algorithm) + Reinforcement learning based on physiological behaviour of pleth.
- Calculation of Features such as:
  - Time-domain: HRV (Heart Rate Variability), mean HR, PTT (Pulse Transit Time)
  - Morphological: Pulse amplitude, rising time, dicrotic notch position
  - ?
(Still Studying)
- Some Type of Causal Modeling Architecture (X features, treatment) → outcome using TarNet
- Purpose of Causal Modeling: Predicting patient deterioration, alarm risk, personalization



## References
- Charlton, P. H., Prada, E. J. A., Mant, J., & Kyriacou, P. A. (2025). The MSPTDfast photoplethysmography beat detection algorithm: design, benchmarking, and open-source distribution. Physiological Measurement. https://doi.org/10.1088/1361-6579/adb89e
- Charlton, P. H. (n.d.). PPG-Beats. https://ppg-beats.readthedocs.io/en/latest/
**AF Annotations**
-  https://figshare.com/articles/dataset/Atrial_Fibrillation_annotations_of_electrocardiogram_from_MIMIC_III_matched_subset/12149091/1 