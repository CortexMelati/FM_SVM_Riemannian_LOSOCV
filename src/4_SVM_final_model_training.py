"""
=============================================================================
4. Final SVM Model Evaluation & Interpretation (LOSOCV - FIXED)
=============================================================================
Overview:
    This unified script replaces the traditional Train/Test split evaluation.
    It performs Leave-One-Subject-Out Cross-Validation (LOSOCV) to rigorously
    evaluate the SVM models on the Master Dataset.

    Crucial Fixes:
    - Hyperparameter tuning uses StratifiedGroupKFold to prevent single-class 
      test-set metric crashing during search.
    - Strict probability index matching to prevent inverted AUROC.
    - Permutation plot uses a vertical density linegraph (Acc on Y, Density on X).
    - ADDED: Calculates in-dataset CV Training Score (Mean ± STD) to assess overfitting.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mne
import sys
import joblib

from pathlib import Path

from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             roc_auc_score, confusion_matrix, brier_score_loss,
                             average_precision_score, balanced_accuracy_score)
from sklearn.utils import shuffle, resample
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, StratifiedGroupKFold

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))
from config import (PROCESSED_DATA_DIR, SVM_FIGURES_DIR, SVM_DATA_DIR, BANDS, CP_FM_DIR, RANDOM_STATE)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
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

def plot_svm_network_map(shap_values_fm, feature_names, target_band):
    mean_abs_shap = np.abs(shap_values_fm).mean(axis=0)
    shap_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_abs_shap}).sort_values(by='Importance', ascending=False)
    shap_df['Node1'] = shap_df['Feature'].apply(lambda x: x.split('-')[0])
    shap_df['Node2'] = shap_df['Feature'].apply(lambda x: x.split('-')[1].split('(')[0])
    
    standard_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(ch_names=standard_19, sfreq=500, ch_types='eeg')
    info.set_montage(montage)

    fig, ax = plt.subplots(figsize=(8, 8))
    mne.viz.plot_sensors(info, show_names=True, axes=ax)
    
    for collection in ax.collections:
        collection.set_sizes([600])
        collection.set_facecolor('white')
        collection.set_edgecolor('#cccccc')
        collection.set_linewidth(1.5)
        
    sensor_offsets = ax.collections[0].get_offsets()
    ch_pos = {ch: (sensor_offsets[i, 0], sensor_offsets[i, 1]) for i, ch in enumerate(info.ch_names)}
    
    # Voorkom zero-division als max_importance 0 is
    max_importance = shap_df['Importance'].max()
    if max_importance == 0: max_importance = 1e-9 
    
    for _, row in shap_df.iterrows():
        try:
            x_coords = [ch_pos[row['Node1']][0], ch_pos[row['Node2']][0]]
            y_coords = [ch_pos[row['Node1']][1], ch_pos[row['Node2']][1]]
            val = row['Importance']
            
            if val >= max_importance * 0.80:
                color, lw = '#FF8C94', 5.0 
            elif val >= max_importance * 0.40:
                color, lw = '#8B4513', 3.5 
            else:
                color, lw = '#228B22', 2.0 
                
            ax.plot(x_coords, y_coords, color=color, linewidth=lw, zorder=0, alpha=0.9)
        except KeyError:
            pass

    ax.set_title(f"Connectivity features associated with fibromyalgia\n({target_band.upper()} Band - SHAP Importance)", fontsize=14, pad=20)
    plt.tight_layout()
    SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SVM_FIGURES_DIR / f"Figure_4_SVM_network_map_{target_band}_LOSOCV.png", dpi=300, transparent=False) 
    plt.close()


def get_bootstrap_ci(y_true, y_pred, metric_func, n_iterations=1000, random_state=RANDOM_STATE):
    scores = []
    y_t, y_p = np.array(y_true), np.array(y_pred)
    for i in range(n_iterations):
        idx = resample(range(len(y_t)), random_state=random_state + i)
        scores.append(metric_func(y_t[idx], y_p[idx]))
    lower, upper = np.percentile(scores, 2.5) * 100, np.percentile(scores, 97.5) * 100
    return f"({lower:.1f} - {upper:.1f})"

def plot_permutation_distribution(permuted_scores, actual_acc, pvalue, target_band):
    print(f"  -> Generating Permutation Test Distribution Plot for {target_band.upper()} band...")
    
    plt.figure(figsize=(8, 6))
    plt.hist(permuted_scores, bins=30, color='#93c59e', edgecolor='black', alpha=0.7, density=True, label='Permuted Scores (Null Distribution)')
    
    # Lijn voor de daadwerkelijke model score
    plt.axvline(actual_acc, color='#d62728', linestyle='dashed', linewidth=2.5, 
                label=f'Actual Score ({actual_acc:.4f})')
    
    # Lijn voor het gemiddelde toevalsniveau
    plt.axvline(np.mean(permuted_scores), color='black', linestyle='dotted', linewidth=2, 
                label=f'Chance Level ({np.mean(permuted_scores):.4f})')

    plt.title(f"Permutation Test Distribution (1000 Iterations)\n({target_band.upper()} Band - p = {pvalue:.4f})", fontsize=12, pad=15)
    plt.xlabel('Balanced Accuracy', fontsize=11)
    plt.ylabel('Density', fontsize=11)
    plt.legend(frameon=True, loc='upper right')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    SVM_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = SVM_FIGURES_DIR / f"Figure_Permutation_Distribution_{target_band}_LOSOCV.png"
    plt.savefig(plot_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def evaluate_all_svm_bands_losocv():
    print("🚀 STARTING STEP 4: COMBINED SVM TRAINING & EVALUATION (LOSOCV)")

    master_path = PROCESSED_DATA_DIR / "final_dataset_master.csv"
    if not master_path.exists():
        sys.exit("🚨 Master dataset not found.")
    master_df = pd.read_csv(master_path)
    
    if 'Condition' in master_df.columns:
        master_df = master_df[master_df['Condition'] == 'EC'].copy()

    y_master = master_df['Target'].values
    groups_master = master_df['Subject'].values

    tsv_path = CP_FM_DIR / "data" / "participants.tsv"
    participants_df = pd.read_csv(tsv_path, sep='\t') if tsv_path.exists() else None
    if participants_df is not None and 'participant_id' in participants_df.columns:
        participants_df['Subject'] = participants_df['participant_id']

    final_results = []
    logo = LeaveOneGroupOut()

    for band_name in BANDS.keys():
        band_name_lower = band_name.lower()
        features_path = SVM_DATA_DIR / f"final_msffs_selected_features_{band_name_lower}.csv"
        
        if not features_path.exists():
            continue
            
        print(f"\n{'='*60}\n📡 PROCESSING BAND: {band_name.upper()}\n{'='*60}")
        selected_features = pd.read_csv(features_path)['Selected_Features'].tolist()
        
        X_master_final = master_df[selected_features]
        scaler = StandardScaler()
        X_master_scaled = pd.DataFrame(scaler.fit_transform(X_master_final), columns=selected_features)
        
        # A. HYPERPARAMETER TUNING VIA 10x REPEATED 5-FOLD STRATIFIED GROUP CV
        print("-> Commencing GridSearchCV for C and gamma optimization (10x Repeated 5-Fold)...")
        
        tuning_cv = []
        for i in range(10):
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + i)
            tuning_cv.extend(list(sgkf.split(X_master_scaled, y_master, groups_master)))
            
        param_grid = {'C': [0.001, 0.1, 1, 10, 100], 'gamma': np.logspace(-3, 1, 40), 'class_weight': ['balanced']}
        
        grid_search = GridSearchCV(
            SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
            param_grid=param_grid, cv=tuning_cv, scoring='balanced_accuracy', n_jobs=-1
        )
        grid_search.fit(X_master_scaled, y_master, groups=groups_master)
        best_params = grid_search.best_params_
        print(f"-> Optimal Parameters Found: {best_params}")

        # B. MANUAL LOSOCV LOOP FOR UNBIASED EVALUATION
        print("-> Running strict LOSOCV evaluation loop...")
        all_preds, all_probs, all_true, all_subjects = [], [], [], []
        train_fold_scores = [] # Nieuwe variabele voor de train scores
        
        for train_idx, test_idx in logo.split(X_master_scaled, y_master, groups_master):
            X_train, y_train = X_master_scaled.iloc[train_idx], y_master[train_idx]
            X_test, y_test = X_master_scaled.iloc[test_idx], y_master[test_idx]
            groups_train = groups_master[train_idx]
            
            # 1. INTERNAL TUNING: Alleen op de X_train van deze fold
            tuning_cv = []
            for i in range(10):
                sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + i)
                tuning_cv.extend(list(sgkf.split(X_train, y_train, groups_train)))
                
            grid_search = GridSearchCV(
                SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
                param_grid=param_grid, cv=tuning_cv, scoring='balanced_accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train, groups=groups_train)
            
            # 2. EVALUATIE: Train model met de interne best_params
            model = SVC(**grid_search.best_params_, probability=True, random_state=RANDOM_STATE)
            model.fit(X_train, y_train)
            
            # --- majority voting ---
            train_preds_epochs = model.predict(X_train)
            df_tr = pd.DataFrame({'Subject': groups_master[train_idx], 'True': y_train, 'Pred': train_preds_epochs})
            df_tr_sub = df_tr.groupby('Subject').agg(T=('True', 'first'), P=('Pred', lambda x: x.mode()[0]))
            train_fold_scores.append(balanced_accuracy_score(df_tr_sub['T'], df_tr_sub['P']))
            
            # Veilig de correcte kans voor de positieve klasse (1) uithalen, voorkomt AUROC 0.0 fouten
            pos_class_idx = np.where(model.classes_ == 1)[0][0]
            
            all_preds.extend(model.predict(X_test))
            all_probs.extend(model.predict_proba(X_test)[:, pos_class_idx])
            all_true.extend(y_test)
            all_subjects.extend(groups_master[test_idx])

        # Bereken de uiteindelijke train mean en standaarddeviatie over de folds heen
        train_mean = np.mean(train_fold_scores)
        train_std = np.std(train_fold_scores)
        print(f"-> CV Training Score: {train_mean:.4f} ± {train_std:.4f}")

        # C. AGGREGATE TO SUBJECT LEVEL (Majority Voting)
        df_preds = pd.DataFrame({'Subject': all_subjects, 'True_Label': all_true, 'Pred_Class': all_preds, 'Pred_Prob': all_probs})
        df_subject = df_preds.groupby('Subject').agg(
            True_Label=('True_Label', 'first'),
            Pred_Class=('Pred_Class', lambda x: x.mode()[0]),
            Consistency=('Pred_Class', lambda x: (x == x.mode()[0]).mean()), # <--- DEZE ONTBRAK
            Pred_Prob=('Pred_Prob', 'mean')
        ).reset_index()

        mean_consistency = df_subject['Consistency'].mean() # <--- DEZE ONTBRAK OOK!

        y_t, y_p, y_prob = df_subject['True_Label'].values, df_subject['Pred_Class'].values, df_subject['Pred_Prob'].values
        
        
        # D. CALCULATE METRICS
        acc = accuracy_score(y_t, y_p)
        bal_acc = balanced_accuracy_score(y_t, y_p) # Extra expliciet voor de tabel
        prec = precision_score(y_t, y_p, zero_division=0)
        rec = recall_score(y_t, y_p, zero_division=0)
        auc = roc_auc_score(y_t, y_prob)
        auprc = average_precision_score(y_t, y_prob) 
        brier = brier_score_loss(y_t, y_prob)
        ece = expected_calibration_error(y_t, y_prob)
        
        # Bereken FPR en FNR
        tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # --- NIEUW: Bereken de 95% CIs via Bootstrapping ---
        ci_bal_acc = get_bootstrap_ci(y_t, y_p, balanced_accuracy_score)
        ci_prec = get_bootstrap_ci(y_t, y_p, lambda yt, yp: precision_score(yt, yp, zero_division=0))
        ci_sens = get_bootstrap_ci(y_t, y_p, lambda yt, yp: recall_score(yt, yp, zero_division=0))

        # --- BEREKEN DE LOSOCV TEST STD ---
        subject_accuracies = (df_subject['True_Label'] == df_subject['Pred_Class']).astype(float)
        test_std = subject_accuracies.std()
        
        print(f"-> LOSOCV Test Validation (Mean ± STD): {acc:.4f} ± {test_std:.4f}")

        # E. PERMUTATION TEST
        n_permutations = 1000
        permuted_scores = [balanced_accuracy_score(shuffle(y_t, random_state=RANDOM_STATE + i), y_p) for i in range(n_permutations)]
        pvalue = (np.sum(np.array(permuted_scores) >= bal_acc) + 1) / (n_permutations + 1)
        plot_permutation_distribution(permuted_scores, bal_acc, pvalue, band_name_lower)
        
        print(f"-> FINAL LOSOCV ACCURACY: {bal_acc:.4f} (p={pvalue:.4f})")


        # F. TRAIN FINAL GLOBAL MODEL FOR SHAP & VISUALIZATIONS
        print("-> Training Final Global Model for Interpretability (SHAP/t-SNE)...")
        global_model = SVC(**best_params, probability=True, random_state=RANDOM_STATE)
        global_model.fit(X_master_scaled, y_master)
        
        artifact = {
            'model': global_model,
            'scaler': scaler,
            'features': selected_features
        }
        joblib.dump(artifact, SVM_DATA_DIR / f"saved_model_{band_name_lower}.pkl")
        print(f"-> Frozen model saved as saved_model_{band_name_lower}.pkl")
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(X_master_scaled)-1), random_state=RANDOM_STATE)
        X_tsne = tsne.fit_transform(X_master_scaled)
        plt.figure(figsize=(8, 6))
        scatter = sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_master, palette={0: '#5c8cbc', 1: '#d62728'}, s=80, alpha=0.8)
        plt.title(f"Data Distribution of Selected Connectivity Features\n({band_name.upper()} Band - t-SNE)", fontsize=14, pad=15)
        plt.legend(handles=scatter.get_legend_handles_labels()[0], labels=['Healthy Control', 'Fibromyalgia'], frameon=True)
        sns.despine()
        plt.tight_layout()
        plt.savefig(SVM_FIGURES_DIR / f"Figure_5_tsne_distribution_{band_name_lower}_LOSOCV.png", dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()

        background = shap.kmeans(X_master_scaled, 10)
        explainer = shap.KernelExplainer(global_model.predict_proba, background)
        np.random.seed(RANDOM_STATE)
        shap_values = explainer.shap_values(X_master_scaled)
        shap_values_fm = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values)
        
        plt.figure(figsize=(10, 8)) 
        shap.summary_plot(shap_values_fm, X_master_scaled, plot_type="bar", show=False)
        ax = plt.gca()
        for patch in ax.patches:
            if patch.get_width() > 0: ax.text(patch.get_width() * 1.01, patch.get_y() + patch.get_height() / 2, f"{patch.get_width():.4f}", va='center')
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.15)
        plt.title(f"{band_name.upper()} Band - Mean Absolute SHAP Values", pad=15)
        plt.tight_layout()
        plt.savefig(SVM_FIGURES_DIR / f"Figure_6A_SHAP_bar_{band_name_lower}_LOSOCV.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8)) 
        shap.summary_plot(shap_values_fm, X_master_scaled, show=False)
        plt.title(f"{band_name.upper()} Band - SHAP Values Summary", pad=15)
        plt.tight_layout()
        plt.savefig(SVM_FIGURES_DIR / f"Figure_6B_SHAP_summary_{band_name_lower}_LOSOCV.png", dpi=300, bbox_inches='tight')
        plt.close()

        plot_svm_network_map(shap_values_fm, selected_features, band_name_lower)

        # G. DEMOGRAPHIC BIAS REPORT
        if participants_df is not None:
            merged_df = pd.merge(df_subject, participants_df, on='Subject', how='inner')
            if not merged_df.empty:
                merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
                merged_df['age_group'] = pd.cut(merged_df['age'], bins=[0, 40, 55, 100], labels=['< 40 years', '40 - 55 years', '> 55 years'])
                bias_results = []
                
                def eval_bias(df_sub, cat, val):
                    if len(df_sub) > 0: bias_results.append({'Category': cat, 'Group': val, 'N': len(df_sub), 'Acc': accuracy_score(df_sub['True_Label'], df_sub['Pred_Class'])})
                
                if 'sex' in merged_df.columns:
                    for s in merged_df['sex'].dropna().unique(): eval_bias(merged_df[merged_df['sex'] == s], 'Sex', 'Female' if s.lower()=='f' else 'Male')
                if 'age_group' in merged_df.columns:
                    for a in merged_df['age_group'].dropna().unique(): eval_bias(merged_df[merged_df['age_group'] == a], 'Age', a)
                
                if bias_results:
                    pd.DataFrame(bias_results).to_csv(SVM_DATA_DIR / f"svm_algorithmic_bias_{band_name_lower}_LOSOCV.csv", index=False)

    
        # H. LOG RESULTS
        opt_params = f"C={best_params['C']}, g={best_params['gamma']:.4f}"        
        final_results.append({
            'Band': band_name.upper(),
            'Optimal_Params': opt_params,
            'Consistency': f"{mean_consistency:.2%}",
            'CV_Training_Score': f"{train_mean:.3f} ± {train_std:.3f}", 
            'Bal_Accuracy': f"{bal_acc:.2%} {ci_bal_acc}",
            'Sensitivity': f"{rec:.2%} {ci_sens}",
            'Precision': f"{prec:.2%} {ci_prec}",
            'FPR': f"{fpr:.2%}",
            'FNR': f"{fnr:.2%}",
            'AUROC': f"{auc:.4f}",
            'AUPRC': f"{auprc:.4f}",
            'Brier': f"{brier:.4f}",
            'ECE': f"{ece:.4f}",
            'Permutation_P': f"{pvalue:.4f}"  
        })
        
        
        # Tekstrapport opslaan (Inclusief CIs)
        report_text = (
            f"====================================================\n"
            f" FINAL TEST SET METRICS SUBJECT - {band_name.upper()} BAND \n"
            f"====================================================\n"
            f"CV Training Score:         {train_mean:.4f} ± {train_std:.4f}\n"
            f"Intra-Subject Consistency: {mean_consistency:.2%}\n"
            f"Balanced Accuracy:         {bal_acc:.4f} {ci_bal_acc}\n"
            f"Precision:                 {prec:.4f} {ci_prec}\n"
            f"Sensitivity (Recall):      {rec:.4f} {ci_sens}\n"
            f"FPR:                       {fpr:.4f}\n"
            f"FNR:                       {fnr:.4f}\n"
            f"ROC-AUC:                   {auc:.4f}\n"
            f"AUPRC:                     {auprc:.4f}\n"
            f"Brier Score:               {brier:.4f}\n"
            f"ECE:                       {ece:.4f}\n"
            f"Permutation P:             {pvalue:.4f}\n"
            f"====================================================\n"
        )
        report_path = SVM_DATA_DIR / f"final_test_metrics_report_{band_name_lower}_LOSOCV.txt"
        with open(report_path, "w") as f:
            f.write(report_text)


    # 3. EXPORT MASTER TABLE
    if final_results:
        results_df = pd.DataFrame(final_results)
        
        # Kolomvolgorde forceren voor een strakke LaTeX tabel
        cols = ['Band', 'Optimal_Params', 'Consistency', 'CV_Training_Score', 'Bal_Accuracy', 'Sensitivity', 'Precision', 'FPR', 'FNR', 'AUROC', 'AUPRC', 'Brier', 'ECE', 'Permutation_P']        
        results_df = results_df[cols]
        
        results_df.to_csv(SVM_DATA_DIR / "final_svm_test_table_LOSOCV.csv", index=False)
        print(f"\n{'='*70}\n🏆 ALL SVM BANDS EVALUATED SUCCESSFULLY (LOSOCV)!\n{'='*70}")
        print(results_df.to_string(index=False))
        print("\n✅ Master table and all figures saved.")

if __name__ == "__main__":
    evaluate_all_svm_bands_losocv()