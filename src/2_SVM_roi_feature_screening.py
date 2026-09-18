"""
=============================================================================
2. ROI Feature Screening (Top 10 Selection)
=============================================================================
Overview:
    This script filters the master dataset strictly to the 9-channel 
    Central ROI and the designated focus band (resulting in 36 potential 
    features). It trains a baseline SVM on the full master dataset and 
    uses SHAP to identify the Top 10 most predictive connectivity features 
    within this restricted space.
    
    The output of this script is fed into the mSFFS pipeline, where 
    strict LOSOCV will be applied.

Execution:
    python 2_SVM_roi_feature_screening.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import mne
import sys
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, 
                    FIGURES_DIR, BEST_CHANNELS_EVALUATE, FOCUS_BAND)

# ==========================================
# 1. SETUP AND FILTERING
# ==========================================
print(f"Starting ROI Feature Screening ({FOCUS_BAND.upper()} Band)...")

# Wijziging: Load the new Master Dataset
master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
if not master_path.exists():
    sys.exit(f"🚨 Error: {master_path.name} not found. Run 2_build_dataset.py first.")

master_df = pd.read_csv(master_path)

y_master = master_df['Target'].values
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_master_full = master_df.drop(columns=[c for c in meta_cols if c in master_df.columns])

# Filter to only include features where BOTH channels are in the ROI and in the Focus Band
roi_features = []
for col in X_master_full.columns:
    if f'({FOCUS_BAND})' in col:
        pair = col.replace(f'({FOCUS_BAND})', '').split('-')
        if pair[0] in BEST_CHANNELS_EVALUATE and pair[1] in BEST_CHANNELS_EVALUATE:
            roi_features.append(col)

X_master_roi = X_master_full[roi_features]
print(f"-> Filtered to {len(roi_features)} features (9 ROI channels, {FOCUS_BAND.upper()} band).")

# ==========================================
# 2. SCALING AND SVM TRAINING
# ==========================================
scaler = StandardScaler()
X_master_scaled = pd.DataFrame(scaler.fit_transform(X_master_roi), columns=X_master_roi.columns)

print("-> Training intermediate SVM for SHAP screening...")
screening_svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
screening_svm.fit(X_master_scaled, y_master)

# ==========================================
# 3. SHAP ANALYSIS
# ==========================================
print("-> Calculating SHAP values for the ROI features...")

background = shap.kmeans(X_master_scaled, 10) 
explainer = shap.KernelExplainer(screening_svm.predict_proba, background)

np.random.seed(RANDOM_STATE)
shap_values = explainer.shap_values(X_master_scaled)

if isinstance(shap_values, list):
    shap_values_fm = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_fm = np.array(shap_values)[:, :, 1]
else:
    shap_values_fm = np.array(shap_values)

mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)

feature_importance = pd.DataFrame({
    'Feature': X_master_scaled.columns,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

top_10_df = feature_importance.head(10)
print(f"\nTOP 10 FEATURES WITHIN ROI ({FOCUS_BAND.upper()} Band):")
print(top_10_df.to_string(index=False))

top_5_df = feature_importance.head(5)

# Save the Top 10 list so the mSFFS script can load it
top_10_path = PROCESSED_DATA_DIR / f"top_10_roi_features_{FOCUS_BAND}.csv"
top_10_df.to_csv(top_10_path, index=False)
print(f"-> Saved Top 10 features to {top_10_path.name}")

# ==========================================
# 4. PLOT TOPOGRAPHICAL NETWORK
# ==========================================
print("\nGenerating Topographical Map for Top 5 Features...")

standard_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
montage = mne.channels.make_standard_montage('standard_1020')
info = mne.create_info(ch_names=standard_19, sfreq=500, ch_types='eeg')
info.set_montage(montage)

fig, ax = plt.subplots(figsize=(8, 8))
mne.viz.plot_sensors(info, show_names=True, axes=ax)

for collection in ax.collections:
    collection.set_sizes([600])
    collection.set_facecolor('white')
    collection.set_edgecolor('#cccccc')
    collection.set_linewidth(1.5)

sensor_offsets = ax.collections[0].get_offsets()
ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}

max_shap = top_5_df['Mean_Abs_SHAP'].max()

for _, row in top_5_df.iterrows():
    feat = row['Feature']
    node1 = feat.split('-')[0]
    node2 = feat.split('-')[1].split('(')[0]
    
    try:
        x_coords = [ch_pos[node1][0], ch_pos[node2][0]]
        y_coords = [ch_pos[node1][1], ch_pos[node2][1]]
        
        val = row['Mean_Abs_SHAP']
        if val >= max_shap * 0.80:       # Top 20% connecties (Roze)
            color, lw = '#FF8C94', 5.0
        elif val >= max_shap * 0.40:     # Top 20-60% connecties (Bruin)
            color, lw = '#8B4513', 3.5
        else:                            # Onderste 40% (Groen)
            color, lw = '#228B22', 2.0
            
        ax.plot(x_coords, y_coords, color=color, linewidth=lw, alpha=0.9, zorder=0)
    except KeyError:
        pass

ax.set_title(f"Top 5 Connectivity Features within ROI\n({FOCUS_BAND.upper()} Band - SHAP Importance)", fontsize=14, pad=20)
plt.tight_layout()

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_path = FIGURES_DIR / f"Figure_Intermediate_Top5_ROI_{FOCUS_BAND}_LOSOCV.png"
plt.savefig(plot_path, dpi=300, transparent=False)
plt.close()

print(f"-> Intermediate plot saved to {plot_path.name}")