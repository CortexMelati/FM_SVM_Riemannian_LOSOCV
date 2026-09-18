"""
=============================================================================
7. RIEMANNIAN CROSS-DOMAIN VALIDATION (Riemannian Alignment)
=============================================================================
Overview:
    Replicates Section 2.7 and 3.4 for the Riemannian model.
    Instead of TrAdaBoost, this script explicitly uses Riemannian Alignment 
    (Fréchet mean re-centering) as described in Equation 2 of the thesis.
    
    METHODOLOGICAL UPDATE: Aligned with the Master Cohort (LOSOCV) 
    preprocessing. The Source Domain alignment is now computed across the 
    entire primary dataset before testing on the Target Domain (NCCP).

Execution:
    python 7_Riemann_cross_domain_RA.py
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import joblib
import mne
import warnings
from tqdm import tqdm
from scipy.linalg import fractional_matrix_power

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from pyriemann.utils.mean import mean_covariance
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, CROSS_TARGET_DATASET, RANDOM_STATE

# =========================================================================
# CUSTOM TRANSFORMERS
# =========================================================================
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
    def __init__(self): 
        pass
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        # We assume X has shape (n_epochs, n_channels, n_channels, n_freqs)
        # Average over the last axis (frequencies)
        return np.mean(X, axis=-1)

class NearestSPD(BaseEstimator, TransformerMixin):
    def __init__(self, reg=1e-6): 
        self.reg = reg
        
    def fit(self, X, y=None): 
        return self
        
    def transform(self, X):
        # Maakt de matrix perfect symmetrisch en positief definiet (SPD)
        X_sym = (X + np.transpose(X, (0, 2, 1))) / 2.0
        eyes = np.tile(np.eye(X.shape[1]), (X.shape[0], 1, 1))
        
        # PATCH VOOR OUDE PICKLE BESTANDEN:
        # Als 'self.reg' niet bestaat (omdat het een oud model is), gebruik dan 1e-6
        reg_value = getattr(self, 'reg', 1e-6)
        
        return X_sym + reg_value * eyes

# THE MAGIC: Implementing Equation 2 from the thesis
class RiemannianAligner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.invsqrt_mean_ = None
        
    def fit(self, X, y=None):
        # 1. Calculate the Fréchet mean of the covariance matrices
        ref_mean = mean_covariance(X, metric='riemann')
        # 2. Calculate M^{-1/2}
        self.invsqrt_mean_ = fractional_matrix_power(ref_mean, -0.5).real
        return self
        
    def transform(self, X):
        # 3. Apply C_aligned = M^{-1/2} * C * M^{-1/2}
        X_aligned = np.zeros_like(X)
        for i in range(len(X)):
            X_aligned[i] = self.invsqrt_mean_ @ X[i] @ self.invsqrt_mean_
        return X_aligned



# =========================================================================

def run_alignment_cross_domain():
    print("🚀 STARTING SCRIPT 7: RIEMANNIAN CROSS-DOMAIN VALIDATION (FRÉCHET ALIGNMENT)")
    
    target_models = [
        "model_riemann_beta_roi_TSSVM_Cov.pkl",
        "model_riemann_delta_roi_TSSVM_Coh.pkl"
    ]
    
    for model_name in target_models:
        model_path = RIEMANN_DATA_DIR / model_name
        if not model_path.exists():
            print(f"\n-> Info: Model {model_name} niet gevonden. Skipping...")
            continue
            
        arch_name = model_path.stem.split("roi_")[-1]
        print(f"\n{'='*70}\n🧠 Evaluating Riemannian Alignment for: {model_name}\n{'='*70}")
            
        artifact = joblib.load(model_path)
        full_pipeline = artifact['model']
        band = artifact['band']
        layout = artifact['layout']
        frozen_svm = full_pipeline.named_steps['svm'] 
        
        # AANGEPAST NAAR MASTER DATASET
        y_source = np.load(RIEMANN_DATA_DIR / "y_master_riemann.npy")
        y_target = np.load(RIEMANN_DATA_DIR / f"target_y_{CROSS_TARGET_DATASET.lower()}.npy")
        groups_target = np.load(RIEMANN_DATA_DIR / f"target_groups_{CROSS_TARGET_DATASET.lower()}.npy")

        if 'Cov' in arch_name:
            # Voor TSSVM_Cov zijn de opgeslagen bestanden AL covariantiematrices.
            # We hebben dus geen extra feature extraction pipeline nodig.
            X_source_raw = np.load(RIEMANN_DATA_DIR / f"covs_master_{band.lower()}_{layout.lower()}.npy")
            X_target_raw = np.load(RIEMANN_DATA_DIR / f"target_covs_{CROSS_TARGET_DATASET.lower()}_{band.capitalize()}_{layout.lower()}.npy")
            C_source = X_source_raw
            C_target = X_target_raw
        else:
            # Voor andere modellen (zoals xDAWN of Coh) moeten we de ruwe signalen nog transformeren.
            try:
                cov_step_idx = [i for i, step in enumerate(full_pipeline.steps) if any(x in step[0] for x in ['cov', 'xdawn', 'coh', 'spd'])][-1]
                cov_pipeline = Pipeline(full_pipeline.steps[:cov_step_idx+1])
            except IndexError:
                sys.exit(f"🚨 Fout: Kon de feature extraction stap niet vinden in pipeline voor {model_name}")
                
            X_source_raw = np.load(RIEMANN_DATA_DIR / "X_master_raw.npy")
            X_target_raw = np.load(RIEMANN_DATA_DIR / f"target_X_{CROSS_TARGET_DATASET.lower()}_raw.npy")
            print("-> Extracting Covariance matrices...")
            C_source = cov_pipeline.transform(X_source_raw)
            C_target = cov_pipeline.transform(X_target_raw)

        # ---------------------------------------------------------
        # SOURCE ALIGNMENT & TRAINING (Done once on Master Cohort)
        # ---------------------------------------------------------
        print("-> Aligning Master Source Domain using its Fréchet Mean...")
        source_aligner = RiemannianAligner()
        C_source_aligned = source_aligner.fit_transform(C_source)
        
        # Project Aligned Source to Tangent Space & Train Global SVM
        ts = TangentSpace(metric='riemann')
        X_source_ts = ts.fit_transform(C_source_aligned)
        
        aligned_svm = SVC(C=frozen_svm.C, kernel=frozen_svm.kernel, class_weight='balanced', random_state=RANDOM_STATE)
        aligned_svm.fit(X_source_ts, y_source)

        # ---------------------------------------------------------
        # TARGET ITERATION (2 to 10 Folds)
        # ---------------------------------------------------------
        max_folds = 10 
        results = []

        print(f"\nRunning iterative testing (2 to {max_folds} Folds)...")
        print(f"{'Folds':<10} | {'Avg Train Subs':<15} | {'Alignment Acc':<15} | {'Direct Acc':<15}")
        print("-" * 65)

        for n_splits in tqdm(range(2, max_folds + 1), desc="Folds Progress", colour='green'):
            cv = StratifiedGroupKFold(n_splits=n_splits)
            alignment_scores, direct_scores, train_subs = [], [], []
            
            for train_idx, test_idx in cv.split(C_target, y_target, groups=groups_target):
                C_tgt_tr, y_tgt_tr = C_target[train_idx], y_target[train_idx]
                C_tgt_te, y_tgt_te = C_target[test_idx], y_target[test_idx]
                
                train_subs.append(len(np.unique(groups_target[train_idx])))
                
                # --- Method 1: DIRECT (Target-Only) ---
                ts_direct = TangentSpace(metric='riemann')
                X_tgt_tr_ts_dir = ts_direct.fit_transform(C_tgt_tr)
                X_tgt_te_ts_dir = ts_direct.transform(C_tgt_te)
                
                d_svm = SVC(C=frozen_svm.C, kernel=frozen_svm.kernel, class_weight='balanced', random_state=RANDOM_STATE)
                d_svm.fit(X_tgt_tr_ts_dir, y_tgt_tr)
                pred_d = d_svm.predict(X_tgt_te_ts_dir)
                
                # --- Method 2: RIEMANNIAN ALIGNMENT (Equation 2) ---
                target_aligner = RiemannianAligner()
                # Find Fréchet mean of TARGET TRAINING data, use it to align both Train & Test
                C_tgt_tr_aligned = target_aligner.fit_transform(C_tgt_tr)
                C_tgt_te_aligned = target_aligner.transform(C_tgt_te)
                
                # Project to Tangent space
                X_tgt_te_ts_align = ts.transform(C_tgt_te_aligned)
                
                # Predict using the SVM trained on the Aligned Source data
                pred_align = aligned_svm.predict(X_tgt_te_ts_align)
                
                # --- Subject-Level Voting ---
                df_fold = pd.DataFrame({'Subject': groups_target[test_idx], 'True': y_tgt_te, 'Dir': pred_d, 'Align': pred_align})
                df_sub = df_fold.groupby('Subject').agg(True_L=('True', 'first'), V_Dir=('Dir', lambda x: x.mode()[0]), V_Al=('Align', lambda x: x.mode()[0])).reset_index()
                
                direct_scores.append(balanced_accuracy_score(df_sub['True_L'], df_sub['V_Dir']))
                alignment_scores.append(balanced_accuracy_score(df_sub['True_L'], df_sub['V_Al']))
                
            tqdm.write(f"{n_splits:<10} | {np.mean(train_subs):<15.1f} | {np.mean(alignment_scores):<15.3f} | {np.mean(direct_scores):<15.3f}")
            results.append({'Total_Folds': n_splits, 'Avg_Train_Subjects': np.mean(train_subs), 'Riemannian_Alignment': np.mean(alignment_scores), 'Direct_Training': np.mean(direct_scores)})
            
        # ---------------------------------------------------------
        # PLOTTING & EXPORT
        # ---------------------------------------------------------
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(RIEMANN_DATA_DIR / f"Table_1_Riemann_Alignment_cross_domain_{band}_{arch_name}.csv", index=False)
            
            plt.figure(figsize=(9, 6))
            plt.scatter(results_df['Avg_Train_Subjects'], results_df['Direct_Training'], color='#5c8cbc', label='Direct Training (Target Only)', s=60, alpha=0.9, edgecolor='white')
            plt.scatter(results_df['Avg_Train_Subjects'], results_df['Riemannian_Alignment'], color='#d62728', label='Riemannian Alignment (Transfer)', s=60, alpha=0.9, edgecolor='white')

            z_dir = np.polyfit(results_df['Avg_Train_Subjects'], results_df['Direct_Training'], 1)
            plt.plot(results_df['Avg_Train_Subjects'], np.poly1d(z_dir)(results_df['Avg_Train_Subjects']), color='gray', lw=2.5, alpha=0.8)

            z_trans = np.polyfit(results_df['Avg_Train_Subjects'], results_df['Riemannian_Alignment'], 1)
            plt.plot(results_df['Avg_Train_Subjects'], np.poly1d(z_trans)(results_df['Avg_Train_Subjects']), color='gray', lw=2.5, alpha=0.8)

            plt.title(f"Riemannian Domain Alignment on {CROSS_TARGET_DATASET} \nArchitecture: {arch_name}", fontsize=13, pad=15)
            plt.xlabel('Mean target training subjects', fontsize=12)
            plt.ylabel('Mean test accuracy', fontsize=12)
            
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.legend(frameon=True, loc='upper left', fontsize=11)
            plt.tight_layout()
            
            fig_name = f"Figure_7_Riemann_Alignment_{band}_{arch_name}.png"
            plt.savefig(RIEMANN_FIGURES_DIR / fig_name, dpi=300)
            plt.close()
            
            print(f"✅ Opgeslagen: Tabel en Plot voor {arch_name}!")

    print("\n✅ SCRIPT 7 (ALIGNMENT) VOLLEDIG AFGEROND.")

if __name__ == "__main__":
    run_alignment_cross_domain()