# EEG-Based Fibromyalgia Classification Pipeline

**Inspired by the paper of Li et al. (2026)**
*Paper DOI:* [10.3389/fpain.2025.1704444](https://doi.org/10.3389/fpain.2025.1704444)

### Datasets

* **Primary Dataset (SRPBG):** [https://osf.io/srpbg/files/osfstorage](https://osf.io/srpbg/files/osfstorage) *(Note: Due to hardware/epoch limitations in external datasets like TDBrain, cross-domain validation is strictly performed via intra-dataset cross-cohort splitting on the SRPBG dataset, leveraging its multi-study composition of FM, CBP, and NCCP cohorts).*

## Overview

This repository contains an advanced machine learning pipeline designed to identify neurophysiological biomarkers for Fibromyalgia (FM) using resting-state EEG data. The methodology replicates the feature extraction and classification framework proposed by Li et al. (2026), utilizing spectral connectivity (Coherence) and an optimized Support Vector Machine (SVM).

To comprehensively validate these findings, the repository extends the original methodology by introducing a parallel **Geometry-Based Riemannian Pipeline** (evaluating Spatial Covariance Matrices via Tangent Space mapping), along with robust modules for  **Algorithmic Fairness (Bias)** ,  **Transfer Learning (TrAdaBoost)** , and  **Post-Hoc Interpretability (SHAP & MNE Topography)** .

To maintain high academic standards and prevent data leakage, the pipeline is controlled by a centralized `config.py` and structured into isolated, modular stages.

---

## Pipeline Architecture

### Phase 0: Configuration (`config.py`)

All global variables, random states, channel selections (10-20 system), and the central ablation switch (`USE_ROI: True/False`) are managed here to guarantee architectural synchronization across all scripts.

### Phase 1: Feature-Based Baseline Pipeline (SVM)

* **Data Preprocessing & Feature Extraction (`preprocess_pipeline.py`):** Applies a 50 Hz notch filter and 0.5–44 Hz bandpass filter. Discards the first 10s of data. Divides data into 30s macro-segments and 1s micro-epochs. Extracts 855 Spectral Coherence features across 5 frequency bands using the multitaper method.
* **Dataset Aggregation & Balancing (`build_dataset.py`):** Aggregates features and implements a strict 80/20 train/test split grouped by `Subject` to prevent data leakage. Dynamically balances the training set via data-density calculation and locks exactly 5 segments per subject for the hold-out test set.
* **1. Global Feature Ranking (`1_SVM_global_feature_ranking.py`):** Evaluates the complete feature space (855 features across all channels and bands) using a rapid baseline SVM to calculate global SHAP values. Outputs **Figure 1** (horizontal SHAP bars of the top 15 features), mathematically justifying the selection of the most predictive frequency band and the central channels.
* **2. ROI Feature Screening (`2_SVM_roi_feature_screening.py`):** Isolates the dataset to the optimal frequency band and the 9 central ROI channels (36 possible functional connections). Conducts a localized SHAP analysis to identify the top 10 connectivity features. Outputs an initial network topography map visualizing these 10 candidate connections.
* **3. Feature Selection (`3_SVM_feature_selection_msffs.py`):** Applies the mSFFS algorithm on the screened ROI data to incrementally build feature subsets (1 to 20 features), evaluating them via StratifiedGroupKFold cross-validation. Generates **Figure 3** (mSFFS curve) to determine the exact inflection point where classification accuracy plateaus, effectively identifying the optimal feature subset size (e.g., the top 5 features).
* **4. Final Model Training (`4_SVM_final_model_training.py`):** Restricts the training data strictly to the optimal feature subset identified by mSFFS. Executes a GridSearchCV to tune the RBF kernel hyperparameters (**$C$** and **$\gamma$**). Validates statistical significance using a 1,000-shuffle permutation test. Freezes and exports the final model and scaler as a `.pkl` artifact. Outputs  **Figure 4** , the definitive topographical brain map displaying the selected biomarkers scaled by importance.
* **5. Model Evaluation & Interpretability (`5_SVM_model_evaluation_and_shap.py`):** Evaluates the frozen model exclusively on the unseen 20% hold-out test set. Calculates definitive clinical metrics (Accuracy, Recall, Brier Score, and Expected Calibration Error). Computes final SHAP values for the selected features. Outputs the Confusion Matrix, **Figure 5** (t-SNE/PCA separation plot), and **Figures 6A & 6B** (SHAP bar and beeswarm plots).

### Phase 2: Geometry-Based Riemannian Pipeline

* **1. Riemann Preprocessing (`1_preprocess_riemann.py`):** Filters the raw EEG data and strictly applies the Eyes-Closed (EC) filter. Generates the spatial covariance matrices required for geometric mapping.
* **2. Riemannian Model Training (`2_SVM_Riemannian_Model_Training.py`):** Trains and optimizes the geometric classifiers via GridSearchCV. Evaluates baseline MDM against the Tangent Space SVM (TS-SVM), selects the best-performing architecture, and freezes the model (`.pkl`).
* **3. Biomarker Mapping (`3_Riemannian_Biomarker_Map.py`):** If TS-SVM is selected, mathematically extracts the tangent space weights and projects them onto a paper-ready 19-channel topographical head map.
* **4. Riemannian Evaluation (`4_Riemannian_Model_Evaluation.py`):** Evaluates the frozen geometric model on the unseen test set. Generates a comprehensive `.txt` report containing Accuracy, Brier Score, ECE, Precision, and Recall, alongside the Confusion Matrix.
* **5. Riemannian Data Distribution (`5_Riemannian_Data_Distribution.py`):** Visualizes the class separability by plotting a t-SNE distribution based specifically on the Riemannian Tangent Space data structure.
* **6. Training Logbook (`6_Riemannian_Training_Logbook.py`):** Automatically generates and logs the hyperparameter tuning process for the geometric pipelines.
* **7. Female Sensitivity Analysis (`7_Riemannian_Female_Sensitivity.py`):** Executes a female-only confounding check on the Riemannian data to verify robustness against demographic variance.

### Phase 3: Clinical Fairness & Generalizability

* **Algorithmic Bias Evaluation (`evaluate_bias.py`):** Merges blind test-set predictions from both the SVM and Riemannian models with metadata (`participants.tsv`) to evaluate hardware-specific and demographic fairness (e.g., accuracy across Sex and Age groups).
* **Cross-Domain Validation (`cross_domain_validation.py`):** Tests hardware-independence by evaluating frozen models on isolated target cohorts. Subsequently applies **TrAdaBoost** (Instance-Weighted Domain Adaptation) to calibrate models against hardware-specific distribution shifts. *(Replicates Paper Figure 7).*

## Methodological Safeguards

* **Zero Data Leakage:** The `StandardScaler` (SVM) and Fréchet Mean reference states (Riemann) are fitted solely on the training data. The 20% hold-out sets are transformed exclusively using these frozen parameters.
* **Blind Demographics:** Demographic data (Age, Sex) are strictly excluded from the training feature space to prevent the algorithms from establishing non-physiological diagnostic shortcuts. Bias is evaluated strictly post-hoc.
* **Artifact Prevention:** By epoching the data prior to concatenation, the pipeline actively prevents phase-discontinuity artifacts that could otherwise lead to severe overfitting in lower frequency bands.
