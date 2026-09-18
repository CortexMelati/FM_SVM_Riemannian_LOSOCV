"""
=============================================================================
7. SVM Female-Only Sensitivity Analysis (Confounding Check)
=============================================================================
Overview:
    This script addresses potential sex-related confounding as detailed by
    Li et al. (2026). It isolates the female subjects within the Master 
    dataset, extracts the frozen optimal features, and re-evaluates the 
    performance to confirm that the model's predictive capability is not 
    driven by sex imbalance.

    METHODOLOGICAL UPDATE: Aligned with the Master Cohort preprocessing.
    Utilizes Leave-One-Subject-Out Cross-Validation (LOSOCV) with 
    subject-level majority voting to compute unbiased Out-Of-Fold metrics.

Execution:
    python 7_SVM_female_sensitivity_analysis.py
=============================================================================
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import joblib
from collections import Counter
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (PROCESSED_DATA_DIR, SVM_DATA_DIR, FOCUS_BAND, 
                    CP_FM_DIR, RANDOM_STATE)

print(f"Starting Female-Only Sensitivity Analysis ({FOCUS_BAND.upper()} Band)...")

# =============================================================================
# 1. LOAD MASTER DATA & FILTER FOR EYES CLOSED (EC)
# =============================================================================
master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
if not master_path.exists():
    print(f"Error: Could not find {master_path.name}. Please run your previous dataset scripts first.")
    sys.exit()

master_df = pd.read_csv(master_path)

if 'Condition' in master_df.columns:
    master_df = master_df[master_df['Condition'] == 'EC'].copy()

# =============================================================================
# 2. MERGE WITH DEMOGRAPHIC METADATA & ISOLATE FEMALES
# =============================================================================
tsv_path = CP_FM_DIR / "data" / "participants.tsv"
if not tsv_path.exists():
    print(f"FATAL ERROR: Cannot find participants.tsv at path:\n{tsv_path}")
    sys.exit()

participants_df = pd.read_csv(tsv_path, sep='\t')

# Match indices exactly matching the fix from Script 5
if 'participant_id' in participants_df.columns:
    participants_df['Subject'] = participants_df['participant_id']

merged_df = pd.merge(master_df, participants_df[['Subject', 'sex']], on='Subject', how='inner')

if merged_df.empty:
    print("Error: Merge failed. Subject IDs between master data and participants.tsv do not match.")
    sys.exit()

# Filter for female participants only ('f' or 'F')
female_df = merged_df[merged_df['sex'].str.lower() == 'f'].copy()
unique_females = female_df['Subject'].nunique()

print(f"-> Total segments in master set: {len(master_df)}")
print(f"-> Isolated female-only subset:    {len(female_df)} segments across {unique_females} unique subjects.")
print("\n🔍 DEMOGRAPHIC VERIFICATION (Sanity Check):")
print(f"-> Genders in this subset: {female_df['sex'].unique()}")
print(f"-> Class distribution (0 = HC, 1 = FM):")
print(female_df['Target'].value_counts().to_string())
print("="*60 + "\n")

y_female = female_df['Target'].values
groups_female = female_df['Subject'].values

# =============================================================================
# 3. LOAD FROZEN ARCHITECTURE (Features & Hyperparameters)
# =============================================================================
model_path = SVM_DATA_DIR / f"saved_model_{FOCUS_BAND}.pkl"
if not model_path.exists():
    print(f"Error: Frozen model {model_path.name} not found. Run Script 4 first.")
    sys.exit()

artifact = joblib.load(model_path)
selected_features = artifact['features']
frozen_svm = artifact['model']

print(f"-> Loaded {len(selected_features)} optimal mSFFS features from frozen artifact.")
print(f"-> Loaded optimized hyperparameters: C={frozen_svm.C}, gamma={frozen_svm.gamma}")

X_female = female_df[selected_features]

# =============================================================================
# 4. SCALING & LOSOCV (Female-Only Space)
# =============================================================================
scaler = StandardScaler()
X_female_scaled = pd.DataFrame(scaler.fit_transform(X_female), columns=selected_features)

logo = LeaveOneGroupOut()

# =============================================================================
# 5. CROSS-VALIDATION EVALUATION (OOF Subject-Level Majority Voting)
# =============================================================================
print(f"-> Running LOSOCV on female subset ({unique_females} iterations)...")

# Initialize a clean SVM model using the exact frozen parameters
sensitivity_svm = SVC(
    kernel='rbf',
    C=frozen_svm.C,
    gamma=frozen_svm.gamma,
    class_weight=frozen_svm.class_weight,
    random_state=RANDOM_STATE
)

y_true_sub = []
y_pred_sub = []

for train_idx, val_idx in tqdm(logo.split(X_female_scaled, y_female, groups=groups_female), total=unique_females, colour='cyan'):
    X_tr, y_tr = X_female_scaled.iloc[train_idx], y_female[train_idx]
    X_val, y_val = X_female_scaled.iloc[val_idx], y_female[val_idx]
    
    sensitivity_svm.fit(X_tr, y_tr)
    preds_epochs = sensitivity_svm.predict(X_val)
    
    # Majority vote per subject
    final_vote = Counter(preds_epochs).most_common(1)[0][0]
    
    y_true_sub.append(y_val[0]) # True label of this subject
    y_pred_sub.append(final_vote)

# Calculate Final OOF Metric
bal_acc = balanced_accuracy_score(y_true_sub, y_pred_sub)

print("\n" + "="*60)
print(" SENSITIVITY ANALYSIS RESULTS (FEMALE-ONLY LOSOCV)")
print("="*60)
print(f"-> OOF Subject-Level Balanced Accuracy: {bal_acc:.4f}")
print("="*60)

# Save results for automated LaTeX text generation
output_df = pd.DataFrame([{
    'Analysis': 'Female-Only Sensitivity (LOSOCV)',
    'Balanced_Accuracy': round(bal_acc, 4),
    'Unique_Subjects': unique_females,
    'Total_Segments': len(female_df)
}])

output_path = SVM_DATA_DIR / f"svm_female_sensitivity_results_{FOCUS_BAND}.csv"
output_df.to_csv(output_path, index=False)
print(f"-> Sensitivity report securely saved to: svm_data/{output_path.name}\n")