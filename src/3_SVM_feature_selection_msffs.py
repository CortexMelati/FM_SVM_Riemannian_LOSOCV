"""
=============================================================================
3. SVM Feature Selection (mSFFS) with LOSOCV
=============================================================================
Overview:
    This script performs Sequential Forward Floating Selection (mSFFS) on the 
    restricted ROI feature space using the Master Dataset. 
    Crucially, it utilizes Leave-One-Subject-Out Cross-Validation (LOSOCV) 
    (LeaveOneGroupOut) to evaluate the feature subsets. This ensures strict 
    subject isolation and is optimal for smaller sample sizes.
    
    It evaluates subsets of increasing sizes (from 1 to 20 features),
    replicates Figure 3, and exports the definitive list of optimal features.

Execution:
    python 3_SVM_feature_selection_msffs.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (RESULTS_DIR, RANDOM_STATE, PROCESSED_DATA_DIR, SVM_DATA_DIR,
                    SVM_FIGURES_DIR, BEST_CHANNELS_EVALUATE, FOCUS_BAND)

# =============================================================================
# PUBLICATION PLOT FUNCTION
# =============================================================================
def plot_msffs_curve(features_count, train_scores, cv_scores, cv_std, ci_margins, target_band):
    plt.figure(figsize=(12, 6))
    x_axis = np.array(features_count)
    
    plt.fill_between(x_axis, cv_scores - ci_margins, cv_scores + ci_margins, 
                 color="#93c59e", alpha=0.4, label='95% Confidence Interval')
    
    plt.plot(x_axis, cv_scores, marker='o', markersize=4, color='#5c8cbc', lw=1.5, label='LOSOCV accuracy')
    plt.plot(x_axis, train_scores, marker='o', markersize=4, color='#fba232', lw=1.5, label='Training accuracy')
    
    for i, (tr, cv, ci) in enumerate(zip(train_scores, cv_scores, ci_margins)):
        plt.text(x_axis[i], tr + 0.002, f"{tr:.3f}", color='#fba232', fontsize=9, ha='center', va='bottom')
        plt.text(x_axis[i], cv - 0.003, f"{cv:.3f}\n(±{ci:.3f})", color='#5c8cbc', fontsize=8, ha='center', va='top')

    plt.title(f"LOSOCV accuracy scores when searching in ROI ({target_band.upper()} band)", fontsize=12, pad=20)
    plt.xlabel('Number of features used', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.xticks(np.arange(min(features_count), max(features_count)+1, 1.0))
    plt.ylim([min(cv_scores - cv_std) - 0.05, 1.05])
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    
    SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = SVM_FIGURES_DIR / f"Figure_3_mSFFS_curve_{target_band}_LOSOCV.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> mSFFS Learning curve saved to {plot_path.name}")

# =============================================================================
# 1. LOAD MASTER DATA & IMPORT SCRIPT 2 FEATURES
# =============================================================================
print(f"Starting mSFFS Feature Selection ({FOCUS_BAND.upper()} Band ROI) with LOSOCV...")

master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
if not master_path.exists():
    sys.exit("🚨 Error: final_dataset_master.csv not found.")
master_df = pd.read_csv(master_path)

y_master = master_df['Target'].values
groups_master = master_df['Subject'].values

# A. Load the Top 10 features discovered in Script 2
top_10_path = PROCESSED_DATA_DIR / f"top_10_roi_features_{FOCUS_BAND}.csv"
if not top_10_path.exists():
    sys.exit(f"Error: Could not find {top_10_path.name}. Please run Script 2 first.")
    
top_10_features = pd.read_csv(top_10_path)['Feature'].tolist()

# B. Get all potential ROI features to fill up the remaining spots
meta_cols = ['Subject', 'Target', 'Condition', 'Segment']
X_master_full = master_df.drop(columns=[c for c in meta_cols if c in master_df.columns])

all_roi_features = []
for col in X_master_full.columns:
    if f'({FOCUS_BAND})' in col:
        pair = col.replace(f'({FOCUS_BAND})', '').split('-')
        if pair[0] in BEST_CHANNELS_EVALUATE and pair[1] in BEST_CHANNELS_EVALUATE:
            all_roi_features.append(col)

# C. Isolate exactly 20 features
remaining_roi = [f for f in all_roi_features if f not in top_10_features]
pool_of_20_features = top_10_features + remaining_roi[:10]

X_master_roi = X_master_full[pool_of_20_features]

print(f"-> Search space strictly constrained to {len(X_master_roi.columns)} features.")

# =============================================================================
# 2. SCALING & LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION
# =============================================================================
scaler = StandardScaler()
X_master_scaled = pd.DataFrame(scaler.fit_transform(X_master_roi), columns=X_master_roi.columns)

logo = LeaveOneGroupOut()
# Maak direct een lijst van alle LOSOCV train/test indexen aan voor mlxtend
cv_splits = list(logo.split(X_master_scaled, y_master, groups=groups_master))
n_splits_for_ci = len(cv_splits)

print(f"-> Created {n_splits_for_ci} LOSOCV folds for strict subject-independent mSFFS scoring.")
# =============================================================================
# 3. mSFFS ALGORITHM
# =============================================================================
print("-> Running mSFFS algorithm (Evaluating subsets from 1 to 20 features)...")

base_svm = SVC(
    kernel='rbf', 
    gamma='scale', 
    class_weight='balanced', 
    random_state=RANDOM_STATE
)

sfs = SFS(
    base_svm, 
    k_features=(1, 20),
    forward=True,
    floating=True,
    scoring='balanced_accuracy', 
    cv=cv_splits,          
    n_jobs=-1              
)

sfs = sfs.fit(X_master_scaled, y_master)
metric_dict = sfs.get_metric_dict()

# =============================================================================
# 4. STATISTICAL EVALUATION
# =============================================================================
print("\n" + "="*85)
print(f"{'k':<3} | {'Added/Changed Feature':<25} | {'Mean Acc':<9} | {'Std Dev':<8} | {'95% CI'}")
print("-" * 85)

f_counts, cv_scores, cv_stds, tr_scores, ci_margins = [], [], [], [], []
stats_results = [] 
max_acc = 0
optimal_k = 1

for k in range(1, len(metric_dict) + 1):
    if k not in metric_dict: continue
    
    if k == 1:
        step_feature = f"+{metric_dict[k]['feature_names'][0]}"
    else:
        prev_set = set(metric_dict[k-1]['feature_names'])
        curr_set = set(metric_dict[k]['feature_names'])
        added = curr_set - prev_set
        removed = prev_set - curr_set
        
        changes = []
        if added: changes.append(f"+{', '.join(added)}")
        if removed: changes.append(f"-{', '.join(removed)}")
        step_feature = " ".join(changes)

    fold_scores = metric_dict[k]['cv_scores']
    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    
    # Calculate CI based on number of subjects (since we do LOSOCV)
    ci_margin = 1.96 * (std_acc / np.sqrt(n_splits_for_ci))
    ci_lower, ci_upper = mean_acc - ci_margin, mean_acc + ci_margin
    ci_str = f"[{ci_lower:.4f} - {ci_upper:.4f}]"

    print(f"{k:<3} | {step_feature:<25} | {mean_acc:.4f}   | {std_acc:.4f}   | {ci_str}")
    
    stats_results.append({
        'k': k,
        'Feature_Change': step_feature,
        'Current_Subset': ", ".join(metric_dict[k]['feature_names']),
        'Mean_Acc': round(mean_acc, 4),
        'Std_Dev': round(std_acc, 4),
        '95_percent_CI': ci_str
    })
    
    f_counts.append(k)
    cv_scores.append(mean_acc)
    cv_stds.append(std_acc)
    ci_margins.append(ci_margin) 
    
    subset = list(metric_dict[k]['feature_names'])
    base_svm.fit(X_master_scaled[subset], y_master)
    tr_scores.append(base_svm.score(X_master_scaled[subset], y_master))
    
    if mean_acc > max_acc:
        max_acc = mean_acc
        optimal_k = k

print("=" * 85)

final_features = list(metric_dict[optimal_k]['feature_names'])
final_acc = metric_dict[optimal_k]['avg_score']

print(f"\nOPTIMAL SUBSET DISCOVERED AT k={optimal_k}:")
print(f"-> Selected Validation Accuracy (LOSOCV): {final_acc:.4f}")
print(f"-> Selected Biomarkers: {', '.join(final_features)}")

# =============================================================================
# 5. PLOT AND SAVE EXPORTS
# =============================================================================
SVM_DATA_DIR.mkdir(parents=True, exist_ok=True)

plot_msffs_curve(f_counts, np.array(tr_scores), np.array(cv_scores), np.array(cv_stds), np.array(ci_margins), FOCUS_BAND)

stats_df = pd.DataFrame(stats_results)
stats_path = SVM_DATA_DIR / f"msffs_statistical_summary_{FOCUS_BAND}_LOSOCV.csv"
stats_df.to_csv(stats_path, index=False)

output_df = pd.DataFrame({'Selected_Features': final_features})
output_path = SVM_DATA_DIR / f"final_msffs_selected_features_{FOCUS_BAND}.csv"
output_df.to_csv(output_path, index=False)
print(f"-> Final feature selection securely saved.")