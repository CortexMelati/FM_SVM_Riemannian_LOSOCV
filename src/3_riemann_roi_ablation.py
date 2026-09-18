"""
=============================================================================
3. RIEMANNIAN ABLATION & METRICS EXTRACTION (LOSOCV)
=============================================================================
Overview:
    Automatically reads the scoreboard from Script 2, identifies the Top 
    performing frequency bands, and tests them using either the 9-channel ROI 
    layout or the 19-channel WHOLE brain layout based on the toggle.
    
    METHODOLOGICAL UPDATE: Now utilizes strict Leave-One-Subject-Out 
    Cross-Validation (LOSOCV) with nested hyperparameter tuning and 
    subject-level majority voting to prevent data leakage and provide 
    an unbiased clinical performance metric. 
    
    NEW: Computes full suite of clinical metrics (AUROC, Brier, ECE, FPR, FNR) 
    directly during the LOSOCV loop and saves them to the comprehensive scoreboard.

Execution:
    python 3_riemann_roi_ablation.py
=============================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
import mne
import warnings
import time
from collections import Counter
from tqdm import tqdm

from pyriemann.estimation import XdawnCovariances, Coherences
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import shuffle
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score, balanced_accuracy_score)

warnings.filterwarnings("ignore", message="DC and Nyquist bins are not defined*")

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, BANDS, BEST_BANDS, RIEMANN_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME, CHANNELS_1020, BEST_CHANNELS_EVALUATE

# =============================================================================
# TOGGLE: TRUE = 19 Channels (WHOLE) | FALSE = 9 Channels (ROI)
# =============================================================================
RUN_AS_WHOLE_BRAIN = False
# =============================================================================

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]
LAYOUT_NAME = 'WHOLE' if RUN_AS_WHOLE_BRAIN else 'ROI'

class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

# --- PyRiemann Compatibility ---
try:
    from pyriemann.preprocessing import NearestSPD
except ImportError:
    try:
        from pyriemann.estimation import NearestSPD
    except ImportError:
        from pyriemann.utils.base import nearest_sym_pos_def
        class NearestSPD(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return nearest_sym_pos_def(X)

class AverageFrequencies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.mean(X, axis=-1) if X.ndim == 4 else X

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_weight = np.sum(in_bin) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def run_ablation():
    total_start_time = time.time()
    
    channel_count = 19 if RUN_AS_WHOLE_BRAIN else 9
    print(f"STARTING SCRIPT 3: ABLATION ({channel_count} CHANNELS - {LAYOUT_NAME}) - LOSOCV MODE")
    
    comprehensive_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv"
    if comprehensive_path.exists():
        df_existing = pd.read_csv(comprehensive_path)
    else:
        df_existing = pd.DataFrame()
    
    best_bands = BEST_BANDS 
    bands_str = ", ".join([b.upper() for b in best_bands])
    print(f"Running {LAYOUT_NAME} Ablation on the following bands: {bands_str}.")

    # Load Master dataset
    X_raw = np.load(RIEMANN_DATA_DIR / "X_master_raw.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_master_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_master_riemann.npy")
    
    svm_param_grid = [
        {'svm__C': [0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['linear']},
        {'svm__C': [0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['rbf'], 'svm__gamma': ['scale', 'auto']}
    ]

    results = []
    architectures = ['TSSVM_Cov', 'TSSVM_Xdawn', 'TSSVM_Coh']
    
    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups))

    for band_name in best_bands:
        band_key = next((k for k in BANDS.keys() if k.lower() == band_name.lower()), None)
        l_freq, h_freq = BANDS[band_key]
        print(f"\n{'='*60}\n ANALYZING: {band_name.upper()} BAND ({LAYOUT_NAME})\n{'='*60}")
        
        cov_file = f"covs_master_{band_name}_whole.npy" if RUN_AS_WHOLE_BRAIN else f"covs_master_{band_name}_roi.npy"
        X_covs = np.load(RIEMANN_DATA_DIR / cov_file)

        for p_name in architectures:
            arch_start_time = time.time()
            print(f"Evaluating Architecture: {p_name} via LOSOCV ({n_subjects} subjects)...")
            
            X_input = X_covs if p_name == 'TSSVM_Cov' else X_raw
            
            y_true_subj = []
            y_pred_subj = []
            y_prob_subj = [] # Added for probability metrics
            
            # Setup Pipeline Features
            if p_name == 'TSSVM_Cov':
                fe_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                
            elif p_name == 'TSSVM_Xdawn':
                fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ))]
                if not RUN_AS_WHOLE_BRAIN:
                    fe_steps.append(('roi', ROIExtractor(ROI_INDICES)))
                fe_steps.extend([
                    ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), 
                    ('ts', TangentSpace(metric='riemann')), 
                    ('scaler', StandardScaler())
                ])
                
            # elif p_name == 'TSSVM_Coh':
            #     fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ))]
            #     if not RUN_AS_WHOLE_BRAIN:
            #         fe_steps.append(('roi', ROIExtractor(ROI_INDICES)))
            #     fe_steps.extend([
            #         ('coh', Coherences(coh='lagged')), 
            #         ('avg_freq', AverageFrequencies()), 
            #         ('spd', NearestSPD()), 
            #         ('ts', TangentSpace(metric='riemann')), 
            #         ('scaler', StandardScaler())
            #     ])
            
            # --- 1. LOSOCV EVALUATION (Unbiased Subject-Level Performance) ---
            for train_idx, val_idx in tqdm(logo.split(X_input, y, groups), total=n_subjects, desc=f"   🔄 {p_name}", leave=False, colour='cyan'):
                fe_pipeline = Pipeline(fe_steps)
                
                # Fit pre-transformation op de N-1 subjects
                X_train_trans = fe_pipeline.fit_transform(X_input[train_idx], y[train_idx])
                X_val_trans = fe_pipeline.transform(X_input[val_idx])
                
                # Nested Hyperparameter tuning op de N-1 subjects (5x Repeated 5-Fold)
                cv_inner = []
                for i in range(5):
                    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + i)
                    cv_inner.extend(list(sgkf.split(X_train_trans, y[train_idx], groups[train_idx])))
                
                # Zorg dat probability=True aan staat!
                model_svm = SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE)
                
                inner_grid = [{'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
                              {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}]
                
                search = GridSearchCV(model_svm, inner_grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
                search.fit(X_train_trans, y[train_idx], groups=groups[train_idx])
                
                # Voorspel klassen en de kans (probability) op Fibromyalgie (klasse 1)
                preds_epochs = search.predict(X_val_trans)
                pos_class_idx = np.where(search.best_estimator_.classes_ == 1)[0][0]
                probs_epochs = search.predict_proba(X_val_trans)[:, pos_class_idx]
                
                final_vote = Counter(preds_epochs).most_common(1)[0][0]
                mean_prob = np.mean(probs_epochs)
                
                y_true_subj.append(y[val_idx][0])
                y_pred_subj.append(final_vote)
                y_prob_subj.append(mean_prob)

            # --- CALCULATE ALL METRICS ---
            y_true_subj = np.array(y_true_subj)
            y_pred_subj = np.array(y_pred_subj)
            y_prob_subj = np.array(y_prob_subj)

            mean_acc = balanced_accuracy_score(y_true_subj, y_pred_subj)
            acc = accuracy_score(y_true_subj, y_pred_subj)
            prec = precision_score(y_true_subj, y_pred_subj, zero_division=0)
            rec = recall_score(y_true_subj, y_pred_subj, zero_division=0)
            auc = roc_auc_score(y_true_subj, y_prob_subj)
            auprc = average_precision_score(y_true_subj, y_prob_subj)
            brier = brier_score_loss(y_true_subj, y_prob_subj)
            ece = expected_calibration_error(y_true_subj, y_prob_subj)

            cm = confusion_matrix(y_true_subj, y_pred_subj)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            else:
                fpr, fnr = 0.0, 0.0

            # Permutation P-Value (Subject Level)
            n_permutations = 1000
            permuted_scores = [balanced_accuracy_score(shuffle(y_true_subj, random_state=RANDOM_STATE + i), y_pred_subj) for i in range(n_permutations)]
            pvalue = (np.sum(np.array(permuted_scores) >= mean_acc) + 1) / (n_permutations + 1)

            arch_time = time.time() - arch_start_time
            print(f"\r    ✅ LOSOCV Mean Bal. Acc: {mean_acc:.4f} (Completed in {arch_time:.1f}s)")

            # --- 2. FINAL MODEL FREEZING (Train on all data for deployment) ---
            print("    -> Freezing final optimal model on full master dataset (10x Repeated 5-Fold)...")
            full_steps = fe_steps.copy()
            full_steps.append(('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE)))
            full_pipeline = Pipeline(full_steps)
            
            cv_final = []
            for i in range(10):
                sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + i)
                cv_final.extend(list(sgkf.split(X_input, y, groups)))
                
            final_search = GridSearchCV(full_pipeline, svm_param_grid, cv=cv_final, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
            final_search.fit(X_input, y, groups=groups)
            
            best_params_log = str(final_search.best_params_)
            clean_params_log = best_params_log.replace("'svm__", "'")
            
            results.append({
                'Band': band_name.upper(), 
                'Layout': LAYOUT_NAME, 
                'Architecture': p_name, 
                'CV_Balanced_Accuracy_Mean': mean_acc, 
                'Sensitivity': rec,
                'Precision': prec,
                'FPR': fpr,
                'FNR': fnr,
                'AUROC': auc,
                'AUPRC': auprc,
                'Brier': brier,
                'ECE': ece,
                'Permutation_P': pvalue,
                'Optimal_Params': clean_params_log
            })
            
            layout_str = 'whole' if RUN_AS_WHOLE_BRAIN else 'roi'
            best_name = f"model_riemann_{band_name}_{layout_str}_{p_name}.pkl"
            
            final_pipe = final_search.best_estimator_
            final_pipe.fit(X_input, y)
            
            joblib.dump({
                'model': final_pipe, 
                'band': band_name, 
                'layout': layout_str, 
                'training_balanced_accuracy': mean_acc,
                'training_std': 0.0 
            }, RIEMANN_DATA_DIR / best_name)
            
            print(f"    -> Frozen {LAYOUT_NAME} model saved successfully.\n")

    # =========================================================================
    # Save and make report
    # =========================================================================
    df_new_run = pd.DataFrame(results)
    
    report_text = "====================================================\n"
    report_text += f" FINAL {LAYOUT_NAME} ABLATION RESULTS (LOSOCV) \n"
    report_text += "====================================================\n\n"
    
    for band_name in best_bands:
        band_rows = df_new_run[df_new_run['Band'] == band_name.upper()].sort_values(by='CV_Balanced_Accuracy_Mean', ascending=False)
        if not band_rows.empty:
            best_row = band_rows.iloc[0]
            report_text += f"🏆 WINNER: {band_name.upper()} BAND ({LAYOUT_NAME})\n"
            report_text += f"Architecture:      {best_row['Architecture']}\n"
            report_text += f"Balanced Accuracy: {best_row['CV_Balanced_Accuracy_Mean']:.4f}\n" 
            report_text += f"Optimal Params:    {best_row['Optimal_Params']}\n"
            report_text += "-"*52 + "\n"

    if not df_existing.empty:
        df_existing = df_existing[~((df_existing['Layout'] == LAYOUT_NAME) & (df_existing['Band'].isin([b.upper() for b in best_bands])))]
        
        if 'CV_Balanced_Accuracy' in df_existing.columns and 'CV_Balanced_Accuracy_Mean' not in df_existing.columns:
            df_existing = df_existing.rename(columns={'CV_Balanced_Accuracy': 'CV_Balanced_Accuracy_Mean'})
            
        df_final = pd.concat([df_existing, df_new_run]).sort_values(by=['Band', 'CV_Balanced_Accuracy_Mean'], ascending=[True, False])
    else:
        df_final = df_new_run.sort_values(by=['Band', 'CV_Balanced_Accuracy_Mean'], ascending=[True, False])

    df_final.to_csv(comprehensive_path, index=False)
    
    total_time = (time.time() - total_start_time) / 60
    print(f"\n{report_text}")
    print(f"Script 3 Complete! Total Execution Time: {total_time:.2f} minutes.")
    print("-> Final Scoreboard updated and ALL evaluated models frozen.")

if __name__ == "__main__":
    run_ablation()