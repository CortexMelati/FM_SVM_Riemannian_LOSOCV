"""
=============================================================================
5. RIEMANNIAN MODEL EVALUATION (LOSOCV - Out-of-Fold Subject Level)
=============================================================================
Overview:
    This script evaluates the optimal Riemannian models using the Out-Of-Fold 
    (OOF) predictions from the Leave-One-Subject-Out Cross-Validation (LOSOCV).
    It rigorously recalculates performance across all frequency bands and 
    layouts (ROI vs WHOLE). 
    
    Majority Voting is applied to group the 1-second epochs back into 
    clinical predictions per subject to compute the final clinical metrics 
    (AUROC, Brier, ECE, Confusion Matrix) and generate the t-SNE plot.

Execution:
    python 5_Riemann_Model_Evaluation.py
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
import joblib
from collections import Counter
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import Covariances, XdawnCovariances, Coherences
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score, balanced_accuracy_score)
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, BANDS, BEST_CHANNELS_EVALUATE, CHANNELS_1020, RANDOM_STATE


class AverageFrequencies(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.mean(X, axis=-1) if X.ndim == 4 else X

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

# --- Custom Transformers ---
class MNEBandPass(BaseEstimator, TransformerMixin):
    def __init__(self, l_freq, h_freq, sfreq=500):
        self.l_freq, self.h_freq, self.sfreq = l_freq, h_freq, sfreq
    def fit(self, X, y=None): return self
    def transform(self, X):
        iir_params = dict(order=4, ftype='butter', output='sos')
        return mne.filter.filter_data(X.astype(np.float64), sfreq=self.sfreq, l_freq=self.l_freq, h_freq=self.h_freq, method='iir', iir_params=iir_params, verbose=False)

class ROIExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, indices): self.indices = indices
    def fit(self, X, y=None): return self
    def transform(self, X): return X[:, self.indices, :]

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_limits[i], bin_limits[i+1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            bin_weight = np.sum(in_bin) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece


def plot_permutation_distribution(permuted_scores, actual_acc, pvalue, target_band):
    print(f"  -> Generating Permutation Test Distribution Plot (KDE) for {target_band.upper()} band...")
    
    plt.figure(figsize=(8, 6))
    
    sns.kdeplot(
        permuted_scores, 
        fill=True, 
        color='#93c59e', 
        alpha=0.6, 
        linewidth=2.5,
        bw_adjust=1.5,
        label='Permuted Scores (Null Distribution)'
    )
    
    plt.axvline(actual_acc, color='#d62728', linestyle='dashed', linewidth=2.5, 
                label=f'Actual Model Score ({actual_acc:.4f})')
    
    plt.axvline(np.mean(permuted_scores), color='black', linestyle='dotted', linewidth=2, 
                label=f'Chance Level (Mean: {np.mean(permuted_scores):.4f})')

    plt.title(f"Permutation Test Distribution (1000 Iterations)\n({target_band.upper()} Band - p = {pvalue:.4f})", fontsize=14, pad=15)
    plt.xlabel('Balanced Accuracy', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    plt.legend(frameon=True, loc='upper left', fontsize=10)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RIEMANN_FIGURES_DIR / f"Figure_Permutation_Distribution_{target_band}.png"
    plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  -> Permutation plot saved to riemann_figures/{plot_path.name}")


def evaluate_riemann_testset():
    print("🚀 STARTING STEP 5: RIEMANNIAN EVALUATION (LOSOCV OOF METRICS)")

    # Load Master Dataset
    X_master_raw = np.load(RIEMANN_DATA_DIR / "X_master_raw.npy")
    y_master = np.load(RIEMANN_DATA_DIR / "y_master_riemann.npy")
    groups_master = np.load(RIEMANN_DATA_DIR / "groups_master_riemann.npy") 
    
    scoreboard_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv"
    if not scoreboard_path.exists():
        sys.exit("🚨 Scoreboard not found! Please run Script 2/3 first.")
    scoreboard = pd.read_csv(scoreboard_path)
    
    if 'Layout' not in scoreboard.columns:
        sys.exit("🚨 Kolom 'Layout' ontbreekt in scoreboard! Voeg deze toe in Script 3.")

    ROI_INDICES = [CHANNELS_1020.index(ch) for ch in BEST_CHANNELS_EVALUATE]
    valid_architectures = ['TSSVM_Cov', 'TSSVM_Xdawn', 'TSSVM_Coh']
    
    logo = LeaveOneGroupOut()
    n_subjects = len(np.unique(groups_master))
    final_results = []
    
    layouts_to_test = ['ROI', 'WHOLE']

    for layout in layouts_to_test:
        for band_name, (l_freq, h_freq) in BANDS.items():
            print(f"\n{'='*60}\n📡 ANALYZING: {band_name.upper()} BAND | LAYOUT: {layout}\n{'='*60}")
            
            band_scores = scoreboard[(scoreboard['Band'] == band_name.upper()) & 
                                     (scoreboard['Layout'] == layout) & 
                                     (scoreboard['Architecture'].isin(valid_architectures))]
                                     
            if band_scores.empty:
                print(f"⚠️ Geen getrainde modellen gevonden voor {band_name.upper()} - {layout}. Skipping...")
                continue
                
            best_row = band_scores.loc[band_scores['CV_Balanced_Accuracy_Mean'].idxmax()]
            arch = best_row['Architecture']
            params = ast.literal_eval(best_row['Optimal_Params'])
            
            print(f"-> Optimal Architecture: {arch}")
            print(f"-> Optimal Params: C={params.get('C', 'N/A')}, Kernel={params.get('kernel', 'N/A')}")
            
            # --- BUILD PIPELINE ---
            steps = [('filter', MNEBandPass(l_freq, h_freq, 500))]
            if layout == 'ROI':
                steps.append(('roi', ROIExtractor(ROI_INDICES)))
            
            if arch == 'TSSVM_Cov':
                steps.extend([('cov', Covariances(estimator='oas')), ('ts', TangentSpace(metric='riemann'))])
            elif arch == 'TSSVM_Xdawn':
                steps.extend([('xdawn', XdawnCovariances(nfilter=6, estimator='oas')), ('ts', TangentSpace(metric='riemann'))])
            elif arch == 'TSSVM_Coh':
                steps.extend([
                    ('coh', Coherences(coh='lagged')), 
                    ('avg_freq', AverageFrequencies()), 
                    ('spd', NearestSPD()), 
                    ('ts', TangentSpace(metric='riemann'))
                ])
                
            steps.extend([
                ('scaler', StandardScaler()),
                ('svm', SVC(C=params.get('C', 1.0), kernel=params.get('kernel', 'linear'), gamma=params.get('gamma', 'scale'), class_weight='balanced', probability=True, random_state=RANDOM_STATE))
            ])
            
            pipeline = Pipeline(steps)
            
            # --- LOSOCV LOOP FOR OUT-OF-FOLD (OOF) PREDICTIONS ---
            print(f"-> Generating OOF predictions via LOSOCV ({n_subjects} subjects)...")
            y_true_sub = []
            y_pred_sub = []
            y_prob_sub = []
            
            for train_idx, test_idx in tqdm(logo.split(X_master_raw, y_master, groups_master), total=n_subjects, leave=False, colour='green'):
                X_tr, y_tr = X_master_raw[train_idx], y_master[train_idx]
                X_te, y_te = X_master_raw[test_idx], y_master[test_idx]
                
                pipeline.fit(X_tr, y_tr)
                
                preds = pipeline.predict(X_te)
                pos_class_idx = np.where(pipeline.classes_ == 1)[0][0]
                probs = pipeline.predict_proba(X_te)[:, pos_class_idx]
                
                # Majority vote per subject
                final_vote = Counter(preds).most_common(1)[0][0]
                mean_prob = np.mean(probs)
                
                y_true_sub.append(y_te[0])
                y_pred_sub.append(final_vote)
                y_prob_sub.append(mean_prob)
                
            y_true_sub = np.array(y_true_sub)
            y_pred_sub = np.array(y_pred_sub)
            y_prob_sub = np.array(y_prob_sub)

            # --- CALCULATE METRICS ON SUBJECT LEVEL ---
            acc = accuracy_score(y_true_sub, y_pred_sub)
            bal_acc = balanced_accuracy_score(y_true_sub, y_pred_sub)
            prec = precision_score(y_true_sub, y_pred_sub, zero_division=0)
            rec = recall_score(y_true_sub, y_pred_sub, zero_division=0)
            auc = roc_auc_score(y_true_sub, y_prob_sub)
            auprc = average_precision_score(y_true_sub, y_prob_sub) 
            brier = brier_score_loss(y_true_sub, y_prob_sub)
            ece = expected_calibration_error(y_true_sub, y_prob_sub)
            
            cm = confusion_matrix(y_true_sub, y_pred_sub)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            else:
                fpr, fnr = 0.0, 0.0

            # --- PERMUTATION TEST ---
            n_permutations = 1000
            permuted_scores = []
            for i in range(n_permutations):
                y_shuffled = shuffle(y_true_sub, random_state=RANDOM_STATE + i)
                score = balanced_accuracy_score(y_shuffled, y_pred_sub)
                permuted_scores.append(score)

            pvalue = (np.sum(np.array(permuted_scores) >= bal_acc) + 1) / (n_permutations + 1)
            print(f"-> Subject-Level Bal. Accuracy: {bal_acc:.4f}")
            print(f"-> Permutation P-value: {pvalue:.4f}")

            plot_permutation_distribution(permuted_scores, bal_acc, pvalue, band_name)
            
            train_mean = best_row['CV_Balanced_Accuracy_Mean']
            
            final_results.append({
                'Band': band_name.upper(),
                'Layout': layout, 
                'Optimal_Architecture': arch,
                'Optimal_Params': f"C={params.get('C', 'N/A')}, {params.get('kernel', 'N/A')}",
                'CV_Training_Score': f"{train_mean:.3f}",
                'Bal_Accuracy': f"{bal_acc:.2%}",
                'Sensitivity': f"{rec:.2%}",
                'Precision': f"{prec:.2%}",
                'FPR': f"{fpr:.2%}", 
                'FNR': f"{fnr:.2%}", 
                'AUPRC': f"{auprc:.4f}",
                'AUROC': f"{auc:.4f}",
                'Brier': f"{brier:.4f}",
                'ECE': f"{ece:.4f}",
                'Permutation_P': f"{pvalue:.4f}"
            })
            
            # --- GENERATE PLOTS ---
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=['Healthy (0)', 'Fibro (1)'], 
                        yticklabels=['Healthy (0)', 'Fibro (1)'],
                        annot_kws={"size": 16})
            plt.title(f'Riemannian FINAL Validation ({arch} - {band_name.upper()} - {layout})\nOOF Subject-Level (Bal. Acc: {bal_acc:.2%})', fontsize=14)
            plt.ylabel('True Clinical Diagnosis', fontsize=12)
            plt.xlabel('Predicted Diagnosis (Majority Vote)', fontsize=12)
            plt.tight_layout()
            
            RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            plot_path = RIEMANN_FIGURES_DIR / f"final_confusion_matrix_riemann_{band_name}_{layout}_{arch}.png"
            plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
            plt.close()

            print(f"  -> Generating t-SNE data distribution for Riemannian {arch} ({layout})...")
            # We fit the pipeline once on the full master dataset to get the spatial transformations for t-SNE
            pipeline.fit(X_master_raw, y_master)
            ts_pipeline = Pipeline(pipeline.steps[:-1]) # Extract everything up to the scaler
            X_master_tangent = ts_pipeline.transform(X_master_raw)
            
            tsne = TSNE(n_components=2, perplexity=min(30, len(X_master_tangent)-1), random_state=42)
            X_tsne_riemann = tsne.fit_transform(X_master_tangent)

            plt.figure(figsize=(8, 6))
            scatter = sns.scatterplot(
                x=X_tsne_riemann[:, 0], y=X_tsne_riemann[:, 1], 
                hue=y_master,
                palette={0: '#5c8cbc', 1: '#d62728'},
                s=80, alpha=0.8, edgecolor='white'
            )
            plt.title(f"Riemannian Tangent Space Distribution\n({band_name.upper()} Band - {layout} Layout)", fontsize=14, pad=15)
            plt.xlabel("t-SNE Dimension 1", fontsize=11)
            plt.ylabel("t-SNE Dimension 2", fontsize=11)

            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            handles, labels = scatter.get_legend_handles_labels()
            plt.legend(handles=handles, labels=['Healthy Control (HC)', 'Fibromyalgia (FM)'], title='Diagnosis', frameon=True)
            plt.tight_layout()
            
            tsne_path = RIEMANN_FIGURES_DIR / f"Figure_5_tsne_riemann_{band_name}_{layout}_{arch}.png"
            plt.savefig(tsne_path, dpi=300, facecolor='white', bbox_inches='tight')
            plt.close()

    # 6. EXPORT MASTER TABLE FOR LATEX
    results_df = pd.DataFrame(final_results)
    csv_path = RIEMANN_DATA_DIR / "final_riemannian_test_table.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*70}\n🏆 ALL BANDS & LAYOUTS EVALUATED SUCCESSFULLY!\n{'='*70}")
    print("Here is your final data for the LaTeX Table 2:\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_riemann_testset()