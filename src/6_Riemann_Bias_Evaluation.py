"""
=============================================================================
6. RIEMANNIAN MODEL BIAS EVALUATION (Demographic Confounding Check)
=============================================================================
Overview:
    This script evaluates the winning Riemannian model (TSSVM_Xdawn)
    for potential demographic bias across age groups and biological sex. 
    
    METHODOLOGICAL UPDATE: To prevent data leakage and evaluate true 
    generalization fairness, this script computes Out-Of-Fold (OOF) 
    predictions using Leave-One-Subject-Out Cross-Validation (LOSOCV) 
    on the Master dataset, before merging with metadata for subgroup analysis.

Execution:
    python 6_Riemann_Bias_Evaluation.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import ast
import mne  
from collections import Counter
from tqdm import tqdm

from sklearn.metrics import accuracy_score, recall_score 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, CP_FM_DIR, BANDS, BEST_CHANNELS_EVALUATE, CHANNELS_1020, RANDOM_STATE

# =========================================================================
# BLAUWDRUKKEN: Nodig om de Xdawn pipeline op te bouwen
# =========================================================================
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X): 
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=dict(order=4, ftype='butter', output='sos'), verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]
# =========================================================================

def evaluate_riemann_bias():
    print("🚀 STARTING STEP 6: RIEMANNIAN DEMOGRAPHIC BIAS EVALUATION (LOSOCV OOF)")

    # # 1. CONFIGURATIE VAN HET TE TESTEN WINNENDE MODEL
    # TARGET_BAND = 'THETA'
    # TARGET_LAYOUT = 'ROI'
    # TARGET_ARCH = 'TSSVM_Xdawn'
    
    # 1. CONFIGURATIE VAN HET TE TESTEN WINNENDE MODEL
    TARGET_BAND = 'BETA'
    TARGET_LAYOUT = 'ROI'
    TARGET_ARCH = 'TSSVM_Cov'
    
    print(f"-> Analyzing bias for: {TARGET_BAND} Band ({TARGET_LAYOUT} Layout) using {TARGET_ARCH}")

    # 2. HAAL DE OPTIMALE PARAMETERS UIT HET SCOREBOARD
    scoreboard_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv"
    if not scoreboard_path.exists():
        sys.exit("🚨 Scoreboard niet gevonden! Draai Script 3 eerst.")
        
    scoreboard = pd.read_csv(scoreboard_path)
    band_scores = scoreboard[(scoreboard['Band'] == TARGET_BAND) & 
                             (scoreboard['Layout'] == TARGET_LAYOUT) & 
                             (scoreboard['Architecture'] == TARGET_ARCH)]
                             
    if band_scores.empty:
        sys.exit(f"🚨 Het model {TARGET_ARCH} voor {TARGET_BAND} is niet gevonden in het scoreboard!")
        
    params = ast.literal_eval(band_scores.iloc[0]['Optimal_Params'])
    print(f"-> Optimal Params Extracted: C={params.get('C', 1.0)}, Kernel={params.get('kernel', 'linear')}")

    # 3. LAAD MASTER DATA EN METADATA
    X_master_path = RIEMANN_DATA_DIR / "X_master_raw.npy"
    y_master_path = RIEMANN_DATA_DIR / "y_master_riemann.npy"
    groups_master_path = RIEMANN_DATA_DIR / "groups_master_riemann.npy"
    tsv_path = CP_FM_DIR / "data" / "participants.tsv"
    
    if not (y_master_path.exists() and X_master_path.exists() and groups_master_path.exists() and tsv_path.exists()):
        sys.exit("🚨 Essentiële masterbestanden of participants.tsv ontbreken.")

    y_master = np.load(y_master_path)
    X_master = np.load(X_master_path)
    groups_master = np.load(groups_master_path)
    participants_df = pd.read_csv(tsv_path, sep='\t')

    # 4. BOUW DE PIPELINE
    l_freq, h_freq = BANDS[TARGET_BAND.capitalize()]
    ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]
    
    pipeline = Pipeline([
        ('filter', MNEBandPass(l_freq, h_freq, 500)),
        ('roi', ROIExtractor(ROI_INDICES)),
        ('xdawn', XdawnCovariances(nfilter=6, estimator='oas')),
        ('ts', TangentSpace(metric='riemann')),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=params.get('C', 1.0), kernel=params.get('kernel', 'linear'), gamma=params.get('gamma', 'scale'), class_weight='balanced', random_state=RANDOM_STATE))
    ])

    # 5. LOSOCV LOOP VOOR UNBIASED (OOF) VOORSPELLINGEN
    print("-> Generating unbiased Out-Of-Fold predictions (LOSOCV)...")
    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups_master))
    
    y_true_sub = []
    y_pred_sub = []
    subject_ids = []
    
    for train_idx, test_idx in tqdm(logo.split(X_master, y_master, groups_master), total=n_subjects, leave=False, colour='magenta'):
        X_tr, y_tr = X_master[train_idx], y_master[train_idx]
        X_te, y_te = X_master[test_idx], y_master[test_idx]
        
        pipeline.fit(X_tr, y_tr)
        preds_epochs = pipeline.predict(X_te)
        
        # Majority vote per subject
        final_vote = Counter(preds_epochs).most_common(1)[0][0]
        
        y_true_sub.append(y_te[0])
        y_pred_sub.append(final_vote)
        subject_ids.append(groups_master[test_idx][0])
        
    df_subject = pd.DataFrame({
        'Subject': subject_ids,
        'True_Label': y_true_sub,
        'Pred_Label': y_pred_sub
    })
    
    df_subject['Is_Correct'] = (df_subject['True_Label'] == df_subject['Pred_Label']).astype(int)

    # 6. MERGE MET DEMOGRAFISCHE DATA
    if 'participant_id' in participants_df.columns:
        participants_df['Subject'] = participants_df['participant_id']

    merged_df = pd.merge(df_subject, participants_df[['Subject', 'sex', 'age']], on='Subject', how='inner')
    
    if merged_df.empty:
        sys.exit("🚨 Merge mislukt. Controleer of de Subject ID's overeenkomen.")

    merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
    merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 40, 55, 100], labels=['< 40 years', '40 - 55 years', '> 55 years'])

    print(f"-> Evaluation matrix built successfully across {len(merged_df)} unique subjects.")

    # 7. BEREKEN ACCURAATHEID EN SENSITIVITY PER SUBGROEP
    bias_results = []
    
    # Biologisch Geslacht
    if 'sex' in merged_df.columns:
        for sex in merged_df['sex'].dropna().unique():
            sub_df = merged_df[merged_df['sex'] == sex]
            if len(sub_df) > 0:
                sex_label = 'Female' if sex.lower() == 'f' else ('Male' if sex.lower() == 'm' else sex.upper())
                acc = accuracy_score(sub_df['True_Label'], sub_df['Pred_Label'])
                sens = recall_score(sub_df['True_Label'], sub_df['Pred_Label'], pos_label=1, zero_division=0)
                
                bias_results.append({'Factor': 'Biological Sex', 'Subgroup': sex_label, 'Accuracy': acc, 'Sensitivity': sens, 'Sample_Size': len(sub_df)})

    # Leeftijdscategorieën
    if 'age_group' in merged_df.columns:
        for age_g in merged_df['age_group'].cat.categories:
            sub_df = merged_df[merged_df['age_group'] == age_g]
            if len(sub_df) > 0:
                acc = accuracy_score(sub_df['True_Label'], sub_df['Pred_Label'])
                sens = recall_score(sub_df['True_Label'], sub_df['Pred_Label'], pos_label=1, zero_division=0)
                
                bias_results.append({'Factor': 'Age Category', 'Subgroup': age_g, 'Accuracy': acc, 'Sensitivity': sens, 'Sample_Size': len(sub_df)})

    bias_df = pd.DataFrame(bias_results)
    
    print("\n🏆 RIEMANNIAN DEMOGRAPHIC PERFORMANCE MATRIX (Ready for LaTeX):")
    print("-" * 75)
    print(bias_df[['Factor', 'Subgroup', 'Accuracy', 'Sensitivity', 'Sample_Size']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("-" * 75)

    bias_csv_path = RIEMANN_DATA_DIR / f"riemann_demographic_bias_report_{TARGET_BAND}_Xdawn.csv"
    bias_df.to_csv(bias_csv_path, index=False, float_format='%.4f')
    
    # 8. VISUALISATIE
    plt.figure(figsize=(8, 5))
    sns.barplot(data=bias_df, x='Subgroup', y='Accuracy', hue='Factor', palette='Oranges_r')
    plt.axhline(0.50, color='gray', linestyle='--', alpha=0.7, label='Chance Level (50%)')
    plt.ylim(0, 1.05)
    plt.ylabel('Out-of-Fold Accuracy', fontsize=12)
    plt.xlabel('Demographic Subgroup', fontsize=12)
    plt.title(f'Riemannian Model Fairness Check ({TARGET_BAND} Band)\nArchitecture: {TARGET_ARCH} (LOSOCV)', fontsize=14, pad=15)
    plt.legend(frameon=True, loc='upper right')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width()/2., height + 0.02, f'{height:.3f}', ha="center", fontsize=10)

    plt.tight_layout()
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RIEMANN_FIGURES_DIR / f"Figure_Riemann_Demographic_Bias_{TARGET_BAND}_xDAWN.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Visual fairness chart saved to: riemann_figures/{plot_path.name}\n")

if __name__ == "__main__":
    evaluate_riemann_bias()