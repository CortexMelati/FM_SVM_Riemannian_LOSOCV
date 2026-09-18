"""
=============================================================================
8. RIEMANNIAN COMPLEXITY CURVE (PCA-BASED FEATURE SELECTION)
=============================================================================
Overview:
    Replicates the visual and analytical intent of Li et al.'s Figure 3.
    Instead of selecting distinct electrode pairs, this script evaluates the 
    number of Principal Components (mathematical features) extracted from the 
    Riemannian Tangent Space needed to reach peak performance.
    
    METHODOLOGICAL UPDATE: Aligned with the Master Cohort preprocessing.
    Utilizes 5-fold Stratified Group CV (to compute confidence intervals) 
    combined with Subject-Level Majority Voting for unbiased metrics.
    
    *Not currently in use for the main thesis analysis.*
    
Execution:
    python 8_R_PCA_Complexity_Curve.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import joblib
import warnings
import mne
from tqdm import tqdm
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, RANDOM_STATE

class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

def plot_complexity_curve():
    print("🚀 STARTING SCRIPT 8: RIEMANNIAN PCA COMPLEXITY CURVE")

    # 1. LAAD HET WINNENDE MODEL & MASTER DATA
    model_name = "model_riemann_Theta_roi_TSSVM_Xdawn.pkl"
    model_path = RIEMANN_DATA_DIR / model_name
    if not model_path.exists():
        sys.exit(f"🚨 Model {model_name} niet gevonden!")
        
    artifact = joblib.load(model_path)
    full_pipeline = artifact['model']
    band = artifact['band']
    
    y_master = np.load(RIEMANN_DATA_DIR / "y_master_riemann.npy")
    groups_master = np.load(RIEMANN_DATA_DIR / "groups_master_riemann.npy")
    X_master = np.load(RIEMANN_DATA_DIR / "X_master_raw.npy")

    # 2. ISOLEER DE TANGENT SPACE PROJECTIE
    fe_pipeline = Pipeline(full_pipeline.steps[:-1])
    frozen_svm = full_pipeline.named_steps['svm'] 
    
    print("-> Projecting raw master data to Tangent Space...")
    X_ts = fe_pipeline.transform(X_master)
    max_features = min(20, X_ts.shape[1]) # We plotten max 20 features, net als Li et al.

    # 3. BEREKEN ACCURAATHEID PER AANTAL COMPONENTEN
    print(f"-> Calculating Subject-Level validation scores for 1 to {max_features} components...")
    
    cv_means, cv_stds, train_means = [], [], []
    feature_range = range(1, max_features + 1)
    
    # 5-fold CV to allow standard deviation calculation for the grey confidence intervals
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    for n_comp in tqdm(feature_range, desc="PCA Components", colour='cyan'):
        pca_svm = Pipeline([
            ('pca', PCA(n_components=n_comp, random_state=RANDOM_STATE)),
            ('svm', SVC(C=frozen_svm.C, kernel=frozen_svm.kernel, class_weight='balanced', random_state=RANDOM_STATE))
        ])
        
        fold_val_scores = []
        fold_train_scores = []
        
        for train_idx, val_idx in cv.split(X_ts, y_master, groups=groups_master):
            X_tr, y_tr, g_tr = X_ts[train_idx], y_master[train_idx], groups_master[train_idx]
            X_val, y_val, g_val = X_ts[val_idx], y_master[val_idx], groups_master[val_idx]
            
            pca_svm.fit(X_tr, y_tr)
            
            # Subject-Level Validation Score
            preds_val = pca_svm.predict(X_val)
            df_val = pd.DataFrame({'Subj': g_val, 'True': y_val, 'Pred': preds_val})
            df_val_sub = df_val.groupby('Subj').agg(T=('True', 'first'), P=('Pred', lambda x: x.mode()[0]))
            fold_val_scores.append(balanced_accuracy_score(df_val_sub['T'], df_val_sub['P']))
            
            # Subject-Level Train Score
            preds_tr = pca_svm.predict(X_tr)
            df_tr = pd.DataFrame({'Subj': g_tr, 'True': y_tr, 'Pred': preds_tr})
            df_tr_sub = df_tr.groupby('Subj').agg(T=('True', 'first'), P=('Pred', lambda x: x.mode()[0]))
            fold_train_scores.append(balanced_accuracy_score(df_tr_sub['T'], df_tr_sub['P']))
            
        cv_means.append(np.mean(fold_val_scores))
        cv_stds.append(np.std(fold_val_scores))
        train_means.append(np.mean(fold_train_scores))

    # 4. PLOT FIGUUR (Exact in de stijl van Li et al.)
    cv_means = np.array(cv_means)
    cv_stds = np.array(cv_stds)
    train_means = np.array(train_means)
    
    plt.figure(figsize=(12, 6))
    
    # Grijze confidence interval (schaduw)
    plt.fill_between(feature_range, cv_means - cv_stds, cv_means + cv_stds, color='#e5ecf6', alpha=0.8, label='_nolegend_')
    
    # Lijnen
    plt.plot(feature_range, cv_means, color='#4b8bbe', marker='.', markersize=8, linewidth=1.5, label='cross-validation accuracy')
    plt.plot(feature_range, train_means, color='#fca311', marker='.', markersize=8, linewidth=1.5, label='training accuracy')
    
    # Annotaties (net als in de paper)
    for i, txt in enumerate(cv_means):
        plt.annotate(f"{txt:.3f}", (feature_range[i], cv_means[i] - 0.008), textcoords="offset points", xytext=(0,-10), ha='center', color='#4b8bbe', fontsize=8)
    for i, txt in enumerate(train_means):
        plt.annotate(f"{txt:.3f}", (feature_range[i], train_means[i] + 0.005), textcoords="offset points", xytext=(0,5), ha='center', color='#fca311', fontsize=8)

    # Opmaak
    plt.title(f"Riemannian Figure 3: Classification accuracy scores when searching in ROI {band.lower()} band.\nGrey area gives the confidence intervals.", loc='left', fontsize=11, pad=15)
    plt.xlabel('number of principal features used', fontsize=10)
    plt.ylabel('balanced accuracy', fontsize=10)
    plt.xticks(feature_range)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(frameon=True, loc='lower right')
    plt.tight_layout()
    
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RIEMANN_FIGURES_DIR / f"Figure_3_Riemann_Complexity_Curve_{band}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\n✅ Plot succesvol gegenereerd en opgeslagen als {plot_path.name}")

if __name__ == "__main__":
    plot_complexity_curve()