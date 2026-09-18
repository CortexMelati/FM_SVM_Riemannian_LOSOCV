"""
=============================================================================
1. PREPROCESS RIEMANN (MASTER DOMAIN & TARGET DOMAINS, DUAL LAYOUT)
=============================================================================
Overview:
    This unified script handles ALL covariance matrix generation.
    
    Part A (Source Domain): Processes the primary dataset into a single 
    Master Tensor for Leave-One-Subject-Out Cross-Validation (LOSOCV).
    
    Part B (Target Domain): Processes the external unseen dataset (e.g., NCCP)
    to ensure it is ready for Cross-Domain Validation (TrAdaBoost).
    
    Generates two spatial layouts: Whole-Brain (19 channels) and Central ROI (9 channels).

Execution:
    python 1_preprocess_riemann.py
=============================================================================
"""

import numpy as np
import pandas as pd
import mne
from pyriemann.estimation import Covariances
from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (PROJECT_ROOT, RESULTS_DIR, PROCESSED_DATA_DIR, SFREQ_MAP, 
                    ACTIVE_DATASET_NAME, BANDS, CHANNELS_1020, BEST_CHANNELS_EVALUATE, 
                    RIEMANN_DATA_DIR, CROSS_TARGET_DATASET)

CONDITION = 'EC'  # or EO
SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
TARGET_SFREQ = SFREQ_MAP.get(CROSS_TARGET_DATASET, 500) 
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

def apply_bandpass_filter(epochs_data, l_freq, h_freq, sfreq=500):
    iir_params = dict(order=4, ftype='butter', output='sos')
    return mne.filter.filter_data(
        epochs_data.astype(np.float64), 
        sfreq=sfreq, 
        l_freq=l_freq, 
        h_freq=h_freq, 
        method='iir', 
        iir_params=iir_params,
        verbose=False
    ) 

def process_source_domain():
    print("🚀 STARTING PART A: SOURCE DOMAIN PREPROCESSING (MASTER LOSOCV)")
    
    master_csv = PROCESSED_DATA_DIR / "final_dataset_master.csv"
    
    if not master_csv.exists():
        raise FileNotFoundError("🚨 Master CSV file missing. Run build_dataset.py first.")
        
    df_master = pd.read_csv(master_csv)
    master_subjects = df_master['Subject'].unique()
    
    file_pattern = f"*_{CONDITION}_cleaned.npy"
    subject_files = list(RESULTS_DIR.rglob(file_pattern))
    
    X_master_list, y_master_list, group_master_list = [], [], []

    for file_path in subject_files:
        subject_id = file_path.name.split('_')[0]
        label = 0 if ("hc" in subject_id.lower() or "control" in str(file_path).lower()) else 1
        
        try:
            data = np.load(file_path)
            # If the subject is in our filtered master cohort, extract the data
            if subject_id in master_subjects:
                target_macro_segments = len(df_master[df_master['Subject'] == subject_id])
                data_trunc = data[:(target_macro_segments * 30)]
                
                X_master_list.append(data_trunc)
                y_master_list.extend([label] * data_trunc.shape[0])
                group_master_list.extend([subject_id] * data_trunc.shape[0])
                
        except Exception as e:
            print(f"  ⚠️ Error processing {file_path.name}: {e}")

    X_master = np.concatenate(X_master_list)
    y_master = np.array(y_master_list)
    groups_master = np.array(group_master_list)

    print(f"  📊 Master Tensor: {X_master.shape}")

    # Save base labels and RAW DATA for the Master cohort
    np.save(RIEMANN_DATA_DIR / "X_master_raw.npy", X_master)
    np.save(RIEMANN_DATA_DIR / "y_master_riemann.npy", y_master)
    np.save(RIEMANN_DATA_DIR / "groups_master_riemann.npy", groups_master)

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"  ⏳ Processing band: {band_name.upper()}...")
        X_filt = apply_bandpass_filter(X_master, l_freq, h_freq, SFREQ)
        
        # Whole Brain
        np.save(RIEMANN_DATA_DIR / f"covs_master_{band_name}_whole.npy", Covariances(estimator='oas').transform(X_filt))
        # ROI
        np.save(RIEMANN_DATA_DIR / f"covs_master_{band_name}_roi.npy", Covariances(estimator='oas').transform(X_filt[:, ROI_INDICES, :]))

    print("✅ Source Preprocessing Complete.\n")

def process_target_domain():
    print(f"🚀 STARTING PART B: TARGET DOMAIN PREPROCESSING ({CROSS_TARGET_DATASET})")
    
    target_csv_path = PROCESSED_DATA_DIR / f"target_domain_{CROSS_TARGET_DATASET.lower()}.csv"
    if not target_csv_path.exists():
        print(f"⚠️ Target CSV niet gevonden. Cross-Domain data wordt overgeslagen.")
        return

    df_target = pd.read_csv(target_csv_path)
    if 'Condition' in df_target.columns:
        df_target = df_target[df_target['Condition'] == 'EC'].copy()

    target_subjects = df_target['Subject'].unique()
    X_target_list, y_target_list, group_target_list = [], [], []
    
    for subject_id in target_subjects:
        subject_files = list(PROJECT_ROOT.rglob(f"{subject_id}*.npy"))
        valid_files = [f for f in subject_files if 'covs' not in f.name and 'y_' not in f.name and 'groups_' not in f.name]
        
        if not valid_files: continue
        file_path = valid_files[0]
        label = df_target[df_target['Subject'] == subject_id]['Target'].values[0]
        
        try:
            data = np.load(file_path)
            target_micro_epochs = len(df_target[df_target['Subject'] == subject_id]) * 30
            data_trunc = data[:target_micro_epochs]
            
            X_target_list.append(data_trunc)
            y_target_list.extend([label] * data_trunc.shape[0])
            group_target_list.extend([subject_id] * data_trunc.shape[0])
        except Exception as e:
            pass

    if not X_target_list:
        print("🚨 FOUT: Geen target data succesvol ingeladen.")
        return

    X_target = np.concatenate(X_target_list)
    np.save(RIEMANN_DATA_DIR / f"target_y_{CROSS_TARGET_DATASET.lower()}.npy", np.array(y_target_list))
    np.save(RIEMANN_DATA_DIR / f"target_groups_{CROSS_TARGET_DATASET.lower()}.npy", np.array(group_target_list))
    np.save(RIEMANN_DATA_DIR / f"target_X_{CROSS_TARGET_DATASET.lower()}_raw.npy", X_target)
    
    print(f"  📊 Target Tensor: {X_target.shape}")

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"  ⏳ Processing target band: {band_name.upper()}...")
        X_filt = apply_bandpass_filter(X_target, l_freq, h_freq, TARGET_SFREQ)
        
        np.save(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band_name}_whole.npy", Covariances(estimator='oas').transform(X_filt))
        np.save(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band_name}_roi.npy", Covariances(estimator='oas').transform(X_filt[:, ROI_INDICES, :]))

    print("✅ Target Preprocessing Complete.")

if __name__ == "__main__":
    RIEMANN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    process_source_domain()
    process_target_domain()