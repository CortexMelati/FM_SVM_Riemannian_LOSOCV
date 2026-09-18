"""
=============================================================================
2. RIEMANNIAN EXPLORATION (19 CHANNELS - WHOLE BRAIN)
=============================================================================
Overview:
    Evaluates all 5 frequency bands using the Whole Brain layout.
    Utilizes a strict Leave-One-Subject-Out Cross-Validation (LOSOCV) 
    framework with nested hyperparameter tuning and subject-level majority voting.
    Saves the best model (.pkl) for each of the 5 bands.

Execution:
    python 2_riemann_whole_brain.py
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import joblib
import mne
import warnings
import time 
from tqdm import tqdm
from collections import Counter

from pyriemann.estimation import Covariances, Coherences, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score

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

warnings.filterwarnings("ignore", message="DC and Nyquist bins are not defined*")

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RANDOM_STATE, BANDS, RIEMANN_DATA_DIR, SVM_DATA_DIR, SFREQ_MAP, ACTIVE_DATASET_NAME

SFREQ = SFREQ_MAP.get(ACTIVE_DATASET_NAME, 500)

class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X):
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class AverageFrequencies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.mean(X, axis=-1) if X.ndim == 4 else X

def run_whole_brain_exploration():
    total_start_time = time.time()
    print("🚀 STARTING SCRIPT 2: FULL BRAIN EXPLORATION (19 CHANNELS - LOSOCV)")
    
    # Laad de Master dataset in in plaats van de oude Train partition
    X_raw = np.load(RIEMANN_DATA_DIR / "X_master_raw.npy")
    y = np.load(RIEMANN_DATA_DIR / "y_master_riemann.npy")
    groups = np.load(RIEMANN_DATA_DIR / "groups_master_riemann.npy")

    svm_param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']}
    ]

    results = []
    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups))

    for band_name, (l_freq, h_freq) in BANDS.items():
        print(f"\n{'='*60}\n📡 ANALYZING: {band_name.upper()} BAND (WHOLE BRAIN)\n{'='*60}")
        band_start_time = time.time()
        
        # Gebruik de Master covariance matrices
        X_covs = np.load(RIEMANN_DATA_DIR / f"covs_master_{band_name}_whole.npy")
        architectures = ['MDM_Cov', 'TSSVM_Cov', 'TSSVM_Xdawn', 'TSSVM_Coh']

        for p_name in architectures:
            arch_start_time = time.time()
            print(f" ⚙️ Evaluating Architecture: {p_name} (LOSOCV over {n_subjects} subjects)...")
            
            X_input = X_covs if 'Cov' in p_name else X_raw
            
            y_true_subj = []
            y_pred_subj = []
            consistency_list = []
            best_params_log = "N/A"
            
            if p_name == 'MDM_Cov':
                pipe_mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
                
                for train_idx, val_idx in tqdm(logo.split(X_input, y, groups), total=n_subjects, desc=f"   🔄 {p_name}", leave=False, colour='cyan'):
                    pipe_mdm.fit(X_input[train_idx], y[train_idx])
                    
                    # Voorspel op alle epochs van het ongeziene subject
                    preds_epochs = pipe_mdm.predict(X_input[val_idx])
                    vote_counts = Counter(preds_epochs)
                    final_vote, vote_freq = vote_counts.most_common(1)[0]
                    
                    # NIEUW: Bereken de consistency voor dit subject
                    consistency = vote_freq / len(preds_epochs)
                    
                    y_true_subj.append(y[val_idx][0])
                    y_pred_subj.append(final_vote)
                    # Bewaar de consistency in een nieuwe lijst (maak bovenaan even 'consistency_list = []' aan)
                    consistency_list.append(consistency)
                    
            else:
                for train_idx, val_idx in tqdm(logo.split(X_input, y, groups), total=n_subjects, desc=f"   🔄 {p_name}", leave=False, colour='cyan'):
                    
                    if p_name == 'TSSVM_Cov':
                        fe_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                    elif p_name == 'TSSVM_Xdawn':
                        fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ)), ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                    # elif p_name == 'TSSVM_Coh':
                    #     fe_steps = [('filter', MNEBandPass(l_freq, h_freq, SFREQ)), ('coh', Coherences(coh='lagged')), ('avg_freq', AverageFrequencies()), ('spd', NearestSPD()), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                    
                    fe_pipeline = Pipeline(fe_steps)
                    
                    # Fit pre-transformation op de N-1 subjects
                    X_train_trans = fe_pipeline.fit_transform(X_input[train_idx], y[train_idx])
                    X_val_trans = fe_pipeline.transform(X_input[val_idx])
                    
                    # Nested Hyperparameter tuning op de N-1 subjects
                    cv_inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
                    model_svm = SVC(class_weight='balanced', random_state=RANDOM_STATE)
                    
                    search = GridSearchCV(model_svm, svm_param_grid, cv=cv_inner, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
                    inner_groups = groups[train_idx]
                    search.fit(X_train_trans, y[train_idx], groups=inner_groups)
                    
                    # Voorspel op alle epochs van het ongeziene subject en pas majority voting toe
                    preds_epochs = search.predict(X_val_trans)
                    final_vote = Counter(preds_epochs).most_common(1)[0][0]
                    
                    y_true_subj.append(y[val_idx][0])
                    y_pred_subj.append(final_vote)
                    best_params_log = str(search.best_params_) # Bewaart de params van de laatste iteratie ter referentie

            # Bereken de uiteindelijke balanced accuracy na alle folds, puur op subject niveau
            mean_acc = balanced_accuracy_score(y_true_subj, y_pred_subj)
            mean_consistency = np.mean(consistency_list) # <--- 3. BEREKEN GEMIDDELDE
            arch_time = time.time() - arch_start_time
            
            print(f"\r    ✅ LOSOCV Subj-Level Bal. Acc: {mean_acc:.4f} (Consistency: {mean_consistency:.2%}) (Completed in {arch_time:.1f}s)")
            
            
            results.append({
                'Band': band_name.upper(), 
                'Layout': 'WHOLE', 
                'Architecture': p_name, 
                'CV_Balanced_Accuracy': mean_acc, 
                'Intra_Subject_Consistency': f"{mean_consistency:.2%}",
                'Optimal_Params': best_params_log
            })
            
        band_time = (time.time() - band_start_time) / 60
        print(f"⏱️ Band {band_name.upper()} finished in {band_time:.2f} minutes.")

    # =========================================================================
    # BEPAAL DE WINNAAR EN SLA OP ALS VOLLEDIGE COMPATIBELE PIPELINE
    # =========================================================================
    df_results = pd.DataFrame(results).sort_values(by='CV_Balanced_Accuracy', ascending=False)
    df_results.to_csv(RIEMANN_DATA_DIR / "scoreboard_whole_brain.csv", index=False)
    
    for band_name in BANDS.keys():
        band_rows = df_results[df_results['Band'] == band_name.upper()]
        if band_rows.empty: continue
        
        best_row = band_rows.iloc[0]
        best_arch = best_row['Architecture']
        
        print(f"-> Freezing final model for {band_name.upper()} ({best_arch}) on Full Master Cohort...")
        
        if best_arch == 'MDM_Cov':
            final_pipe = MDM(metric=dict(mean='riemann', distance='riemann'))
            X_final_input = np.load(RIEMANN_DATA_DIR / f"covs_master_{band_name}_whole.npy")
        else:
            import ast
            p_dict = ast.literal_eval(best_row['Optimal_Params']) if "{" in best_row['Optimal_Params'] else {}
            
            if best_arch == 'TSSVM_Cov':
                final_steps = [('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                X_final_input = np.load(RIEMANN_DATA_DIR / f"covs_master_{band_name}_whole.npy")
            elif best_arch == 'TSSVM_Xdawn':
                final_steps = [('filter', MNEBandPass(BANDS[band_name][0], BANDS[band_name][1], SFREQ)), ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
                X_final_input = X_raw
            # elif best_arch == 'TSSVM_Coh':
            #     final_steps = [('filter', MNEBandPass(BANDS[band_name][0], BANDS[band_name][1], SFREQ)), ('coh', Coherences(coh='lagged')), ('avg_freq', AverageFrequencies()), ('spd', NearestSPD()), ('ts', TangentSpace(metric='riemann')), ('scaler', StandardScaler())]
            #     X_final_input = X_raw
                
            final_steps.append(('svm', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE, **p_dict)))
            final_pipe = Pipeline(final_steps)
            
        final_pipe.fit(X_final_input, y)
        best_name = f"model_riemann_{band_name}_whole_{best_arch}.pkl"
        joblib.dump({'model': final_pipe, 'band': band_name, 'layout': 'whole', 'training_balanced_accuracy': best_row['CV_Balanced_Accuracy']}, SVM_DATA_DIR / best_name)

    total_time = (time.time() - total_start_time) / 60
    print(f"\n✅ Script 2 Complete! Total Execution Time: {total_time:.2f} minutes.")
    print("Whole Brain scoreboard and optimized LOSOCV models saved.")

if __name__ == "__main__":
    run_whole_brain_exploration()