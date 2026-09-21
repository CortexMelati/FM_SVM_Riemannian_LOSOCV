"""
=============================================================================
SCRIPT 3.5: TARGETED RIEMANNIAN METRICS EXTRACTOR (TABLE 12)


    Adheres strictly to the methodology text:
    - Leave-One-Subject-Out Cross-Validation (LOSOCV) outer loop.
    - 5x repeated 5-fold Stratified Group CV inner loop for hyperparameter tuning.

=============================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import mne
import warnings
import time
from collections import Counter
from tqdm import tqdm

from pyriemann.estimation import Covariances, Coherences, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample, shuffle
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score, balanced_accuracy_score)

warnings.filterwarnings("ignore")

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, BANDS, RIEMANN_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME, CHANNELS_1020, BEST_CHANNELS_EVALUATE

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)
ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]

# =============================================================================
# TRANSFORMERS & HELPERS
# =============================================================================
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

class AverageFrequencies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.mean(X, axis=-1) if X.ndim == 4 else X

try:
    from pyriemann.preprocessing import NearestSPD
except ImportError:
    from pyriemann.utils.base import nearest_sym_pos_def
    class NearestSPD(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None): return self
        def transform(self, X): return nearest_sym_pos_def(X)

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

def get_bootstrap_ci(y_true, y_pred, metric_func, n_iterations=1000, random_state=RANDOM_STATE):
    scores = []
    y_t, y_p = np.array(y_true), np.array(y_pred)
    for i in range(n_iterations):
        idx = resample(range(len(y_t)), random_state=random_state + i)
        scores.append(metric_func(y_t[idx], y_p[idx]))
    lower, upper = np.percentile(scores, 2.5) * 100, np.percentile(scores, 97.5) * 100
    return f"({lower:.1f} - {upper:.1f})"

# =============================================================================
# TARGETED EVALUATION LOOP
# =============================================================================
def run_table_12_metrics():
    total_start = time.time()
    print("🚀 STARTING SCRIPT 3.5: STRICT METHODOLOGY TABLE 12 METRICS EXTRACTOR")
    
    X_raw = np.load(RIEMANN_DATA_DIR / "X_master_raw.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_master_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_master_riemann.npy")
    
    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups))
    
    # De drie geselecteerde modellen voor de thesis:
    targets = [
        {'band': 'BETA', 'arch': 'TSSVM_Cov'},
        # {'band': 'DELTA', 'arch': 'TSSVM_Coh'}, # takes 54 hours to process
        # {'band': 'ALPHA', 'arch': 'TSSVM_Xdawn'} # not included in the paper, added for completeness
    ]
    
    results_list = []
    
    for target in targets:
        band_name, arch_name = target['band'], target['arch']
        band_key = next((k for k in BANDS.keys() if k.upper() == band_name.upper()), None)
        l_freq, h_freq = BANDS[band_key]
        
        print(f"\n{'='*60}\n📡 EXTRACTING: {band_name} BAND | {arch_name} (ROI)\n{'='*60}")
        
        if arch_name == 'TSSVM_Cov':
            cov_file = f"covs_master_{band_name.lower()}_roi.npy"
            X_input = np.load(RIEMANN_DATA_DIR / cov_file)
            fe_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
        elif arch_name == 'TSSVM_Coh':
            X_input = X_raw
            fe_steps = [
                ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
                ('roi', ROIExtractor(ROI_INDICES)),
                ('coh', Coherences(coh='lagged')), 
                ('avg_freq', AverageFrequencies()), 
                ('spd', NearestSPD()), 
                ('ts', TangentSpace(metric='riemann')), 
                ('scaler', StandardScaler())
            ]
        elif arch_name == 'TSSVM_Xdawn':
            X_input = X_raw
            fe_steps = [
                ('filter', MNEBandPass(l_freq, h_freq, SFREQ)),
                ('roi', ROIExtractor(ROI_INDICES)),
                ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), 
                ('ts', TangentSpace(metric='riemann')), 
                ('scaler', StandardScaler())
            ]
            
        y_true_subj, y_pred_subj, y_prob_subj, consistency_subj, cv_train_scores = [], [], [], [], []
        
        for train_idx, val_idx in tqdm(logo.split(X_input, y, groups), total=n_subjects, desc=f"   🔄 LOSOCV", leave=False, colour='green'):
            
            # 1. Pipeline samenvoegen (Data Leakage Fix)
            full_steps = fe_steps + [('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE))]
            full_pipeline = Pipeline(full_steps)
            
            # 2. STRICT METHODOLOGY: 5x Repeated 5-Fold Inner Tuning
            cv_inner = []
            for i in range(5):
                sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + i)
                cv_inner.extend(list(sgkf.split(X_input[train_idx], y[train_idx], groups[train_idx])))
            
            inner_grid = [
                {'svm__C': [0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['linear']},
                {'svm__C': [0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['rbf'], 'svm__gamma': ['scale', 'auto']}
            ]
            
            # 3. GridSearchCV aanroepen
            search = GridSearchCV(full_pipeline, inner_grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
            search.fit(X_input[train_idx], y[train_idx], groups=groups[train_idx])
            
            # Sla de beste inner-CV score op voor de \pm in de tabel
            cv_train_scores.append(search.best_score_)
            
            # 4. Voorspellen op de ruwe ongeziene LOSOCV data
            preds_epochs = search.predict(X_input[val_idx])
            pos_class_idx = np.where(search.best_estimator_.classes_ == 1)[0][0]
            probs_epochs = search.predict_proba(X_input[val_idx])[:, pos_class_idx]
            
            # Subject-level aggregatie & Consistency
            vote_counts = Counter(preds_epochs)
            final_vote, vote_freq = vote_counts.most_common(1)[0]
            consistency = vote_freq / len(preds_epochs)
            mean_prob = np.mean(probs_epochs)
            
            y_true_subj.append(y[val_idx][0])
            y_pred_subj.append(final_vote)
            y_prob_subj.append(mean_prob)
            consistency_subj.append(consistency)
            
        # --- CALCULATE FINAL SUBJECT-LEVEL METRICS ---
        y_true_subj, y_pred_subj, y_prob_subj = np.array(y_true_subj), np.array(y_pred_subj), np.array(y_prob_subj)
        
        acc = accuracy_score(y_true_subj, y_pred_subj)
        bal_acc = balanced_accuracy_score(y_true_subj, y_pred_subj)
        prec = precision_score(y_true_subj, y_pred_subj, zero_division=0)
        rec = recall_score(y_true_subj, y_pred_subj, zero_division=0)
        auc = roc_auc_score(y_true_subj, y_prob_subj)
        brier = brier_score_loss(y_true_subj, y_prob_subj)
        ece = expected_calibration_error(y_true_subj, y_prob_subj)
        mean_cons = np.mean(consistency_subj)
        
        # Bereken de gemiddelde CV Training score met standaarddeviatie (\pm)
        mean_cv = np.mean(cv_train_scores)
        std_cv = np.std(cv_train_scores)
        cv_score_str = f"{mean_cv:.3f} ± {std_cv:.3f}"
        
        cm = confusion_matrix(y_true_subj, y_pred_subj)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            fpr, fnr = 0.0, 0.0

        # Bootstrapped CI's (Voor de haakjes in de tabel)
        ci_bal_acc = get_bootstrap_ci(y_true_subj, y_pred_subj, balanced_accuracy_score)
        ci_prec = get_bootstrap_ci(y_true_subj, y_pred_subj, lambda yt, yp: precision_score(yt, yp, zero_division=0))
        ci_rec = get_bootstrap_ci(y_true_subj, y_pred_subj, lambda yt, yp: recall_score(yt, yp, zero_division=0))

        # Permutation P-Value
        n_permutations = 1000
        permuted_scores = [balanced_accuracy_score(shuffle(y_true_subj, random_state=RANDOM_STATE + i), y_pred_subj) for i in range(n_permutations)]
        pvalue = (np.sum(np.array(permuted_scores) >= bal_acc) + 1) / (n_permutations + 1)
        
        print(f"\n   ✅ RESULTS FOR {band_name} ({arch_name}):")
        print(f"   CV Training : {cv_score_str}")
        print(f"   Bal. Acc    : {bal_acc:.2%} {ci_bal_acc}")
        print(f"   Consistency : {mean_cons:.2%}")
        
        results_list.append({
            'Band': band_name,
            'Architecture': arch_name,
            'CV_Training_Score': cv_score_str,
            'Balanced_Accuracy': f"{bal_acc*100:.2f} {ci_bal_acc}",
            'Intra_Subj_Consistency': f"{mean_cons:.2%}",
            'Sensitivity': f"{rec*100:.2f} {ci_rec}",
            'Precision': f"{prec*100:.2f} {ci_prec}",
            'FPR': f"{fpr*100:.2f}",
            'FNR': f"{fnr*100:.2f}",
            'AUROC': f"{auc:.4f}",
            'Brier_Score': f"{brier:.4f}",
            'ECE': f"{ece:.4f}",
            'Permutation_P': f"{pvalue:.4f}"
        })

        df_results = pd.DataFrame(results_list)
        save_path = RIEMANN_DATA_DIR / "table_12_riemann_metrics.csv"
        df_results.to_csv(save_path, index=False)

    time_spent = (time.time() - total_start) / 60
    print(f"🎉 Script Complete! Total Time: {time_spent:.1f} minutes.")
    print(f"💾 All results ready for LaTeX saved to: {save_path.name}")

if __name__ == "__main__":
    run_table_12_metrics()