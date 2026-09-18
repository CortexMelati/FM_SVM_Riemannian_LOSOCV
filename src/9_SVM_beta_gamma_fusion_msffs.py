"""
=============================================================================
9. CROSS-FREQUENCY FUSION (Dynamic mSFFS with LOSOCV)
=============================================================================
Overview:
    This script dynamically reads your top 2 bands from BEST_BANDS in config.py.
    It combines their Top 10 features into a single search space of 20 features 
    and runs the mSFFS algorithm using Leave-One-Subject-Out Cross-Validation 
    (LOSOCV) to test for true complementarity.

Execution:
    python 9_SVM_cross_frequency_fusion_msffs.py
=============================================================================
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# --- CONFIG ROUTING ---
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, PROCESSED_DATA_DIR, BEST_BANDS

if len(BEST_BANDS) < 2:
    sys.exit("🚨 Fout: Zet tenminste 2 banden in 'BEST_BANDS' (config.py) voor de fusion!")

b1, b2 = BEST_BANDS[0].lower(), BEST_BANDS[1].lower()
print(f"🚀 STARTING CROSS-FREQUENCY FUSION EXPERIMENT ({b1.upper()} + {b2.upper()}) [LOSOCV]")

# =============================================================================
# 1. LOAD MASTER DATA & DYNAMIC FEATURES
# =============================================================================
master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
if not master_path.exists():
    sys.exit("🚨 Error: Master dataset not found. Run build_dataset.py first.")
    
master_df = pd.read_csv(master_path)
y_master, groups_master = master_df['Target'].values, master_df['Subject'].values

try:
    top_10_1 = pd.read_csv(PROCESSED_DATA_DIR / f"top_10_roi_features_{b1}.csv")['Feature'].tolist()
    top_10_2 = pd.read_csv(PROCESSED_DATA_DIR / f"top_10_roi_features_{b2}.csv")['Feature'].tolist()
except FileNotFoundError:
    sys.exit(f"🚨 Fout: Zorg dat Script 2 voor zowel {b1.upper()} als {b2.upper()} is gedraaid!")

# Schaal uitsluitend de 20 benodigde features in één snelle stap
X_master_scaled = pd.DataFrame(StandardScaler().fit_transform(master_df[top_10_1 + top_10_2]), columns=top_10_1 + top_10_2)

# =============================================================================
# 2. LOSOCV SETUP & mSFFS ALGORITHM
# =============================================================================
n_folds = 10
cv_strategy = StratifiedGroupKFold(n_splits=n_folds)
cv_splits = list(cv_strategy.split(X_master_scaled, y_master, groups=groups_master))

print(f"-> Created {n_folds} cross-validation folds (StratifiedGroupKFold) for stable fusion scoring.")
print(f"-> Running mSFFS on the combined {len(X_master_scaled.columns)} features...")

base_svm = SVC(kernel='rbf', gamma='scale', class_weight='balanced', random_state=RANDOM_STATE)

sfs = SFS(base_svm, k_features=(1, 20), forward=True, floating=True, scoring='balanced_accuracy', cv=cv_splits, n_jobs=-1)
sfs = sfs.fit(X_master_scaled, y_master)

# =============================================================================
# 3. EVALUATE RESULTS
# =============================================================================
metric_dict = sfs.get_metric_dict()

# Vind automatisch de 'k' met de allerhoogste gemiddelde LOSOCV score
best_k = max(metric_dict.keys(), key=lambda k: np.mean(metric_dict[k]['cv_scores']))
final_features = list(metric_dict[best_k]['feature_names'])

print("\n" + "="*70)
print(f"🏆 FUSION RESULTS (Optimal subset at k={best_k})")
print("="*70)
print(f"-> Max Validation Accuracy (LOSOCV): {np.mean(metric_dict[best_k]['cv_scores']):.4f}")
print(f"-> Selected Biomarkers: {', '.join(final_features)}")

# Check welke banden het model daadwerkelijk heeft behouden
has_b1 = any(f'({b1})' in f for f in final_features)
has_b2 = any(f'({b2})' in f for f in final_features)

print("\n💡 CONCLUSIE:")
if has_b1 and has_b2:
    print(f"YES! Het algoritme combineert {b1.upper()} en {b2.upper()}. Ze zijn complementair!")
else:
    winner = b1.upper() if has_b1 else b2.upper()
    print(f"NOPE. Het algoritme weigert te mixen en kiest uitsluitend {winner} features.")
print("="*70)