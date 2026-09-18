"""
Central Configuration for Thesis EEG Pipeline.
Aligned with the methodology of Li et al. (2026).

config.py
"""

import os
from pathlib import Path


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# This file is located in: .../FM_SVM_RIEMANNIAN/src
CURRENT_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT is one directory up: .../FM_SVM_RIEMANNIAN
PROJECT_ROOT = CURRENT_DIR.parent

# DATA_ROOT is two directories up (Thesis) and then to Data: .../Thesis/Data
DATA_ROOT = Path.home() / "Documents" / "Thesis" / "Data"

# Input Directories
FM_DIR = DATA_ROOT / "FM_EO_dataset"           # Epoched itRS dataset
CP_FM_DIR = DATA_ROOT / "CP_FM_dataset"        # PainMunich dataset
TDBRAIN_DIR = DATA_ROOT / "TDBRAIN-dataset"    # TDBRAIN dataset
CHRONIC_PAIN_DIR = DATA_ROOT / "Chronicpainset" # Chronic Pain dataset
# NOT_USABLE_DIR = DATA_ROOT / "OSF_mj9xr_notusable" # Ignored by the pipeline

# ==========================================
# ABLATION SWITCH (Centralized) Not in use
# ==========================================
# True  = Benchmark mode (9 Central Channels - prevents EMG artifacts)
# False = Exploratory mode (All 19 Channels)
USE_ROI = False
PREFIX = "ROI_" if USE_ROI else "ALL_"

ACTIVE_DATASET_NAME = CP_FM_DIR.name

# Output Directories (dynamisch gerouteerd per dataset)
RESULTS_DIR = PROJECT_ROOT / "results" / ACTIVE_DATASET_NAME
FIGURES_DIR = RESULTS_DIR / "figures"
PROCESSED_DATA_DIR = RESULTS_DIR / "processed_data" 

# riemannian model info
RIEMANN_DATA_DIR = PROCESSED_DATA_DIR / "riemann_data"
RIEMANN_FIGURES_DIR = FIGURES_DIR / "riemann_figures"

# SVM model info
SVM_DATA_DIR = PROCESSED_DATA_DIR / "svm_data"
SVM_FIGURES_DIR = FIGURES_DIR / "svm_figures"


# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RIEMANN_DATA_DIR.mkdir(parents=True, exist_ok=True)
RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
SVM_DATA_DIR.mkdir(parents=True, exist_ok=True)
SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# POST-HOC EVALUATION & CROSS-DOMAIN SETTINGS
# ==============================================================================
# The primary frequency band used for SHAP, Bias Evaluation, and Cross Validation
FOCUS_BAND = 'beta' # change to the best band from 1_SVM_feature_ranking  **Gamma or Beta
BEST_BANDS = ['beta', 'delta'] #whole brain = theta and gamma, ROI = Theta and alpha


# Internal Cross-Domain Validation cohorts (matched with participants.tsv 'study' column)
CROSS_SOURCE_DATASET = "FM"             # Primary cohort for Train/Test
CROSS_TARGET_DATASET = "NCCP"           # Isolated cohort for Transfer Learning/Direct training
# CROSS_TARGET_DATASET = ["NCCP", "CBP"]  # Isolated cohort for Transfer Learning/Direct training option, will try later outside of thesis scope

# ==========================================
# 2. PREPROCESSING & TIMING PARAMETERS (Li et al., 2026)
# ==========================================
# Native sampling rates per hardware system (linked to the respective directory names)
SFREQ_MAP = {
    "FM_EO_dataset": 250,      # Native itRS data
    "CP_FM_dataset": 500,      # PainMunich after downsampling (1000Hz -> 500Hz)
    "TDBRAIN-dataset": 500,    # Native Mitsar system
    "Chronicpainset": 500      # Native continuous system
}

# Substrings in filenames used to automatically assign target labels 
# (0 = Healthy Control, 1 = Patient/Chronic Pain/Fibromyalgia)
LABEL_MAPPING = {
    'Healthy_0': ['FMHC', 'NCCPHC', 'HC'], 
    'Patient_1': ['FMPA', 'CBPPA', 'NCCPPA', 'PA'] 
}


# Filter specifications according to the reference paper
FILTER_HP = 0.5           # High-pass filter cutoff (Hz)
FILTER_LP = 44.0          # Low-pass filter cutoff (Hz)
NOTCH_FREQ = 50.0         # Line noise notch filter (Hz)


# Segmentation windows for CONTINUOUS data (TDBRAIN, Chronicpainset, CP_FM_dataset)
TRUNCATE_TIME = 10.0      # Initial duration to discard for continuous data (seconds)
SEGMENT_LENGTH = 30.0     # Macro-segments for continuous datasets (seconds)
EPOCH_LENGTH = 1.0        # Micro-epochs for connectivity or covariance analysis (seconds)


# Segmentation windows for PRE-EPOCHED itRS data (FM_EO_dataset)
ITRS_TMIN = 0.0           # Start point of the itRS trial
ITRS_TMAX = 1.348         # Exactly 337 samples at 250Hz (approx. 1350 ms)

# ==============================================================================
# 3. CHANNEL CONFIGURATION (Standard International 10-20)
# ==============================================================================
# The exact 19 channels utilized in the core feature space of the paper
CHANNELS_1020 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',   # T3 -> T7, T4 -> T8
    'P7', 'P3', 'Pz', 'P4', 'P8',   # T5 -> P7, T6 -> P8
    'O1', 'O2'
]

# Mapping dictionary to synchronize legacy channel names with
# modern MNE standard montages (e.g., 'standard_1020')
CHANNEL_RENAMING_MAP = {
    'T3': 'T7', 'T4': 'T8',
    'T5': 'P7', 'T6': 'P8'
}


BEST_CHANNELS_EVALUATE = [
    'F3', 'Fz', 'F4',
    'C3', 'Cz', 'C4',
    'P3', 'Pz', 'P4',
]

# ==============================================================================
# 4. FREQUENCY BANDS OF INTEREST
# ==============================================================================
# Strictly defined spectral boundaries corresponding to Equation 1
BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),    
    'Beta':  (12, 30),
    'Gamma': (30, 40)     # capped at 40 as stated in the paper from li et al. 
}

# Multitaper resolutie voor experimentele doeleinden
MT_BANDWIDTH = 4.0 # 4 = standard, 2 = more precise (more noise), 
                    # 8  = smoother (less resolution)




# ==============================================================================
# 6. MACHINE LEARNING & VISUALIZATION METRICS
# ==============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2 # paper uses 0.1 but our dataset is limited
MIN_AGE = 18.0


# Documented core features identified via mSFFS for reference verification
# amend after rolling 2_SVM_roi_feature_screening
OUR_TOP_5_FEATURES = [
    ('Fz', 'Cz'), ('Pz', 'P4'), ('Fz', 'C3'), ('Cz', 'P4'), ('Cz', 'Pz')
]



# Consistent color scheme for plots
COLORS = {
    'Healthy': '#1f77b4',      # Blue
    'ChronicPain': '#d62728',  # Red
    'FM': '#9467bd',           # Purple
    'Unknown': '#7f7f7f'       # Grey
}

if __name__ == "__main__":
    print(f"✅ Configuration Loaded.")
    print(f"📂 Config Location: {CURRENT_DIR}")
    print(f"📂 Project Root:    {PROJECT_ROOT}")
    print(f"📂 Data Root:       {DATA_ROOT}")
    print(f"📂 Results Dir:     {RESULTS_DIR}\n")
    
    datasets = [
        ("FM_EO_dataset (itRS)", FM_DIR), 
        ("TDBRAIN-dataset (Cont)", TDBRAIN_DIR), 
        ("Chronicpainset (Cont)", CHRONIC_PAIN_DIR),
        ("CP_FM_dataset (Cont)", CP_FM_DIR)
    ]
    
    for name, directory in datasets:
        if directory.exists():
            print(f"   -> 📂 {name} path found! ✅")
        else:
            print(f"   -> ❌ {name} path NOT found at: {directory}")
            
    if (RESULTS_DIR / "final_dataset.csv").exists():
        print(f"   -> final_dataset.csv found! ✅")
    else:
        print(f"   -> ⚠️ final_dataset.csv not yet generated in the results folder.")
        
