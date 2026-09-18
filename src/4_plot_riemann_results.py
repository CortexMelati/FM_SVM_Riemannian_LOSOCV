"""
=============================================================================
4. RIEMANNIAN BIOMARKER MAP (TOPOGRAPHY)
=============================================================================
Overview:
    This script generates a top 5 physiological network map for the winning 
    frequency bands (dynamically read from the ROI ablation scoreboard).
    It fits a Surrogate Linear Tangent Space SVM on the Master Cohort strictly 
    for spatial interpretability.
    
python 4_plot_riemann_results.py
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from pyriemann.tangentspace import TangentSpace
from sklearn.svm import SVC
from pathlib import Path
import sys

# Paden instellen
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RIEMANN_DATA_DIR, RIEMANN_FIGURES_DIR, BEST_CHANNELS_EVALUATE

def plot_surrogate_riemannian_weights():
    print("🚀 STARTING SCRIPT 4: DYNAMIC TOPOGRAPHICAL NETWORK MAPPING")
    
    # 1. LEES DE WINNENDE BANDEN UIT SCRIPT 3
    scoreboard_path = RIEMANN_DATA_DIR / "riemann_comprehensive_scoreboard.csv"
    if not scoreboard_path.exists():
        sys.exit(f"🚨 Scoreboard niet gevonden! Draai Script 3 eerst.")
        
    df_all = pd.read_csv(scoreboard_path)
    
    # Filter specifiek op de ROI resultaten
    df_roi = df_all[df_all['Layout'] == 'ROI'].copy()
    
    # Pak unieke banden die geëvalueerd zijn op ROI niveau
    winning_bands = df_roi['Band'].unique().tolist()
    
    if not winning_bands:
        sys.exit("🚨 Geen winnende banden gevonden in de ROI resultaten.")

    # 2. RECONSTRUEER DE TANGENT SPACE INDEXERING
    roi_channels = BEST_CHANNELS_EVALUATE
    n_channels = len(roi_channels)
    pair_map = []
    
    # PyRiemann vectoriseert: eerst diagonaal, dan off-diagonals per rij
    for i in range(n_channels):
        for j in range(i, n_channels):
            pair_map.append((roi_channels[i], roi_channels[j]))

    # AANGEPAST NAAR MASTER DATASET
    y_path = RIEMANN_DATA_DIR / "y_master_riemann.npy"
    if not y_path.exists():
        sys.exit(f"🚨 Kon de labels niet vinden: {y_path}")
    y_master = np.load(y_path)

    # 3. LOOP OVER DE WINNENDE BANDEN EN TEKEN EEN KAART PER BAND
    for band_name in winning_bands:
        print(f"\n{'='*60}\n🧠 GENERATING MAP FOR: {band_name} BAND\n{'='*60}")
        
        # Omdat de band in je scoreboard als UPPERCASE staat, formatten we hem even
        band_file_name = band_name.lower()
        
        # AANGEPAST NAAR MASTER COVARIANCES
        covs_path = RIEMANN_DATA_DIR / f"covs_master_{band_file_name}_roi.npy"
        
        if not covs_path.exists():
            print(f"⚠️ Covariantiematrices voor {band_name} niet gevonden. Wordt overgeslagen.")
            continue
            
        X_cov_master = np.load(covs_path)
        
        # --- TRAIN HET LINEAIRE SURROGATE MODEL ---
        print(f"-> Fitting Linear Surrogate SVM op de {band_name} Master Tangent Space...")
        ts = TangentSpace(metric='riemann')
        X_ts = ts.fit_transform(X_cov_master)
        
        clf = SVC(kernel='linear', C=1.0)
        clf.fit(X_ts, y_master)
        svm_coefs = clf.coef_[0]

        # --- KOPPEL EN FILTER DE GEWICHTEN ---
        weights_df = pd.DataFrame({
            'Node1': [p[0] for p in pair_map],
            'Node2': [p[1] for p in pair_map],
            'Weight': np.abs(svm_coefs)
        })
        
        # Filter kanaal-met-zichzelf eruit
        weights_df = weights_df[weights_df['Node1'] != weights_df['Node2']]
        top_5 = weights_df.sort_values(by='Weight', ascending=False).head(5)
        
        log_text = f"Top 5 Riemannian Connecties (Linear Surrogate) - {band_name} Band\n"
        log_text += "="*65 + "\n"
        
        for _, row in top_5.iterrows():
            line = f"   {row['Node1']:<3} - {row['Node2']:<3} | Gewicht: {row['Weight']:.4f}\n"
            print(line, end="")
            log_text += line

        RIEMANN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        txt_save_path = RIEMANN_FIGURES_DIR / f"Riemann_Network_{band_name}_Surrogate_Weights.txt"
        with open(txt_save_path, "w") as f:
            f.write(log_text)
            
        # --- MNE TOPOGRAFIE TEKENEN ---
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

        max_weight = top_5['Weight'].max()
        top_5['Scaled_Importance'] = (top_5['Weight'] / max_weight) * 2.5

        for _, row in top_5.iterrows():
            try:
                x_coords = [ch_pos[row['Node1']][0], ch_pos[row['Node2']][0]]
                y_coords = [ch_pos[row['Node1']][1], ch_pos[row['Node2']][1]]
                
                scaled_val = row['Scaled_Importance']
                if scaled_val >= 2.0:
                    color, lw = '#FF8C94', 5.0  # Roze (Belangrijkst)
                elif scaled_val >= 1.0:
                    color, lw = '#8B4513', 3.5  # Bruin
                else:
                    color, lw = '#228B22', 2.0  # Groen
                    
                ax.plot(x_coords, y_coords, color=color, linewidth=lw, alpha=0.9, zorder=0)
            except KeyError:
                pass

        ax.set_title(f"Riemannian TS-SVM Connectivity\n({band_name} Band - Linear Surrogate)", fontsize=14, pad=20)
        
        # --- NIEUW: Voeg handmatig een legenda toe voor de lijnkleuren/diktes ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='#FF8C94', lw=5.0, label='Top 20% Impact'),
            Line2D([0], [0], color='#8B4513', lw=3.5, label='Top 20-60% Impact'),
            Line2D([0], [0], color='#228B22', lw=2.0, label='Bottom 40% Impact')
        ]
        ax.legend(handles=legend_elements, loc='lower left', title="Mathematical Vector Importance", fontsize=10)

        plt.tight_layout()
        
        save_path = RIEMANN_FIGURES_DIR / f"Figure_Riemann_Network_{band_name}_Surrogate.png"
        plt.savefig(save_path, dpi=300, transparent=False)
        plt.close()
        print(f"✅ Hersenkaart opgeslagen: {save_path.name}")

if __name__ == "__main__":
    plot_surrogate_riemannian_weights()