"""
=============================================================================
2. DATASET AGGREGATION FOR LOSOCV (Leave-One-Subject-Out)
=============================================================================
Overview:
    Aggregates feature files, identifies the study cohort via participants.tsv,
    and creates a single Master Dataset for the primary cohort.
    The Train/Test split is completely removed here, as the LOSOCV 
    framework will handle dynamic subject-level splitting during model training.
    
    The target domain (e.g., 'NCCP') is still saved separately.
    
Execution:
    python 2_build_dataset.py
=============================================================================
"""

import os
import glob
import pandas as pd
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, LABEL_MAPPING, RANDOM_STATE, 
                    PROCESSED_DATA_DIR, CROSS_SOURCE_DATASET, 
                    CROSS_TARGET_DATASET, CP_FM_DIR)

print("Starting Dataset Aggregation for LOSOCV...")

# =============================================================================
# 1. LOAD & FILTER FEATURES
# =============================================================================
file_pattern = os.path.join(RESULTS_DIR, "**", "*_features.csv")
feature_files = glob.glob(file_pattern, recursive=True)

if not feature_files:
    sys.exit("Error: No *_features.csv files found.")

all_data = [pd.read_csv(file) for file in feature_files]
master_df = pd.concat(all_data, ignore_index=True)

# --- EC FILTERING ---
if 'Condition' in master_df.columns:
    initial_rows = len(master_df)
    master_df = master_df[master_df['Condition'] == 'EC'].copy()
    print(f"-> Filtered to EC only: removed {initial_rows - len(master_df)} non-EC rows.")

def assign_label(subject_id):
    subject_upper = str(subject_id).upper()
    for label_key, identifiers in LABEL_MAPPING.items():
        for identifier in identifiers:
            if identifier in subject_upper:
                return int(label_key.split('_')[1])
    return None

master_df['Target'] = master_df['Subject'].apply(assign_label)
master_df = master_df.dropna(subset=['Target'])
master_df['Target'] = master_df['Target'].astype(int)

# =============================================================================
# 2. MERGE WITH METADATA TO PREVENT DATA MIXING
# =============================================================================
tsv_path = CP_FM_DIR / "data" / "participants.tsv" 
if not tsv_path.exists():
    sys.exit(f"FATAL ERROR: Cannot find participants.tsv at path:\n{tsv_path}")

participants_df = pd.read_csv(tsv_path, sep='\t')
if 'participant_id' in participants_df.columns:
    participants_df['Subject'] = participants_df['participant_id']

merged_df = pd.merge(master_df, participants_df[['Subject', 'study']], on='Subject', how='inner')
if merged_df.empty:
    sys.exit("FATAL ERROR: Merge resulted in 0 rows! Check file names and TSV IDs.")

# =============================================================================
# 3. ISOLATE COHORTS
# =============================================================================
source_cohort_name = CROSS_SOURCE_DATASET
target_cohort_name = CROSS_TARGET_DATASET

source_df = merged_df[merged_df['study'] == source_cohort_name].copy().drop(columns=['study'])
target_df = merged_df[merged_df['study'] == target_cohort_name].copy().drop(columns=['study'])

print("\nDATA SEPARATION COMPLETE:")
print(f"   -> Primary Cohort ({source_cohort_name}): {source_df['Subject'].nunique()} subjects isolated.")
print(f"   -> Target Cohort ({target_cohort_name}): {target_df['Subject'].nunique()} subjects isolated.")

# Save Target Domain
target_path = PROCESSED_DATA_DIR / f"target_domain_{target_cohort_name.lower()}.csv"
target_df.to_csv(target_path, index=False)

# =============================================================================
# 4. CAP SEGMENTS FOR MASTER DATASET
# =============================================================================
# Limit every subject to a maximum of 5 segments (to ensure equal representation)
sampled_master_data = []
for subject, group in source_df.groupby('Subject'):
    sampled_master_data.append(group.sort_values('Segment').head(5))

master_df_final = pd.concat(sampled_master_data).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# =============================================================================
# 5. SAVE FINAL MASTER DATASET
# =============================================================================
master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
master_df_final.to_csv(master_path, index=False)

print("\nMASTER DATASET CREATION SUCCESSFUL (LOSOCV Ready)")
print(f"Master set ({source_cohort_name}): {master_path.name} | Rows: {len(master_df_final)} | Unique Subjects: {master_df_final['Subject'].nunique()}")
print(f"Target ({target_cohort_name}): {target_path.name} | Rows: {len(target_df)}")