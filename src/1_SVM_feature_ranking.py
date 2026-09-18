"""
=============================================================================
1. Initial Global Feature Ranking (LOSOCV Pipeline)
=============================================================================
This script trains a baseline exploratory model on ALL 855 features 
(all bands, all channels) using the full master dataset.
The goal is solely to generate an initial SHAP-ranking to justify the 
choice for specific ROI channels and frequency bands in later chapters.

Note: As this is an exploratory global consensus, it utilizes the full 
master dataset. Strict Leave-One-Subject-Out Cross-Validation (LOSOCV) 
is applied in the subsequent feature selection and evaluation scripts.

Execution:
    python 1_SVM_feature_ranking.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
os.environ["OMP_NUM_THREADS"] = "1"

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, FIGURES_DIR

print("Starting Global Feature Ranking (Exploratory)...")

# 1. Load the full MASTER set
master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
if not master_path.exists():
    sys.exit(f"🚨 Master dataset not found at {master_path}. Run build_dataset.py first.")

master_df = pd.read_csv(master_path)

y_master = master_df['Target'].values
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_master_raw = master_df.drop(columns=[c for c in meta_cols if c in master_df.columns])

print(f"-> Master dataset loaded: {master_df['Subject'].nunique()} subjects, {len(master_df)} segments.")
print(f"-> Number of features loaded: {X_master_raw.shape[1]} (This should be ~855)")

# 2. Scaling
scaler = StandardScaler()
X_master_scaled = pd.DataFrame(scaler.fit_transform(X_master_raw), columns=X_master_raw.columns)

# 3. Train a quick baseline SVM on all features
print("-> Training exploratory baseline SVM on the full feature space...")
global_svm = SVC(kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
global_svm.fit(X_master_scaled, y_master)

# 4. Calculate SHAP Values
print("-> Calculating SHAP values across 855 features...")
# We use K-means (k=10) to create a background distribution to halve the computation time
background = shap.kmeans(X_master_scaled, 10) 
explainer = shap.KernelExplainer(global_svm.predict_proba, background)

np.random.seed(RANDOM_STATE)

# We sample the dataset to calculate the impact
shap_values = explainer.shap_values(X_master_scaled)

if isinstance(shap_values, list):
    shap_values_fm = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_values_fm = np.array(shap_values)[:, :, 1] # Extract only class 1
else:
    shap_values_fm = np.array(shap_values)

# 5. Extract the Top 10 Features
mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)
feature_names = X_master_scaled.columns
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

top_10_features = feature_importance.head(10)

print("\nTOP 10 GLOBAL FEATURES:")
print(top_10_features.to_string(index=False))

# 6. Custom Horizontal Bar Plot (Figure 1 replication with data labels)
plt.figure(figsize=(12, 8))

# Sort ascending purely for plotting (so the highest value is at the top)
plot_df = top_10_features.sort_values(by='Mean_Abs_SHAP', ascending=True)

# Create the horizontal bars
bars = plt.barh(plot_df['Feature'], plot_df['Mean_Abs_SHAP'], color='#1f77b4', height=0.6)

# Add the numerical labels slightly to the right of each bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001,  
             bar.get_y() + bar.get_height() / 2, 
             f"{width:.4f}", 
             ha='left', va='center', fontsize=10, color='black')

# Styling to match typical academic plots
plt.xlabel("Mean |SHAP value|", fontsize=12)
plt.ylabel("Connectivity Feature", fontsize=12)
plt.title("Exploratory Global Feature Ranking", fontsize=14, pad=15)

# Remove top and right spines for a cleaner look
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Extend x-axis slightly so the text labels don't get cut off
current_xlim = ax.get_xlim()
ax.set_xlim(current_xlim[0], current_xlim[1] * 1.15) 

plt.tight_layout()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plot_path = FIGURES_DIR / "Figure_1_Global_SHAP_Ranking_LOSOCV.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ====================================================================
# Channel Importance ("Without the connections")
# ====================================================================
print("\nCalculating individual channel importance (Node Importance)...")
channel_shap = {}

for index, row in feature_importance.iterrows():
    feat = row['Feature'] 
    val = row['Mean_Abs_SHAP']
    
    # Extract the channels
    channels_part = feat.split('(')[0]
    ch1, ch2 = channels_part.split('-')
    
    # Add the SHAP value to both channels
    channel_shap[ch1] = channel_shap.get(ch1, 0) + val
    channel_shap[ch2] = channel_shap.get(ch2, 0) + val

node_importance_df = pd.DataFrame(list(channel_shap.items()), columns=['Channel', 'Total_SHAP'])
node_importance_df = node_importance_df.sort_values(by='Total_SHAP', ascending=False)

print("\nTOP 5 MOST IMPORTANT INDIVIDUAL CHANNELS:")
print(node_importance_df.head(5).to_string(index=False))

print(f"\nAnalysis complete. Plot saved to {plot_path.name}")