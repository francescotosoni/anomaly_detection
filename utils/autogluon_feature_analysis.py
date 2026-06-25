import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from autogluon.tabular import TabularDataset, TabularPredictor
from utils.evaluate_metrics import (
    FEATURES, get_anomaly_files, extract_anomalous_windows
)

def main():
    print("="*80)
    print("  Feature Importance Analysis: AutoGluon Permutation Importance")
    print("="*80)
    
    scaler = joblib.load("data/scaler.pkl")
    
    print(f"\n[1] Caricamento Dati Normali e Anomali...")
    X_test_normal = np.load("data/X_test_normal.npy")
    
    X_list = []
    y_list = []
    
    # Use the last timestep of each normal window
    X_list.append(X_test_normal[:, -1, :])
    y_list.append(np.zeros(len(X_test_normal)))
    
    categories = ["change_weight", "drop_object", "force_stop", "outside_box", "push", "scale_speed", "tcp_wrench"]
    for cat in categories:
        files = get_anomaly_files(cat)
        for f in files:
            try:
                df = pd.read_csv(f, delimiter=' ')
                windows, window_labels = extract_anomalous_windows(df, scaler, file_path=f, labeling_mode="tight")
                
                if len(windows) == 0:
                    continue
                        
                X_list.append(windows[:, -1, :])
                y_list.append(window_labels)
            except Exception as e:
                pass
                
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    print(f"\n[2] Creazione Dataset AutoGluon...")
    print(f"    Totale campioni: {len(X_all)}, Anomalie: {np.sum(y_all)}")
    
    # Crea un DataFrame Pandas
    df_data = pd.DataFrame(X_all, columns=FEATURES)
    df_data['label'] = y_all.astype(int)
    
    # Autogluon Dataset
    train_data = TabularDataset(df_data)
    
    print("\n[3] Training AutoGluon (Time limit: 60s per modello veloce)...")
    # Impostiamo un time_limit e presets veloci per non farti aspettare ore
    predictor = TabularPredictor(label='label', eval_metric='f1').fit(
        train_data, 
        time_limit=120, 
        presets='medium_quality',
        verbosity=2
    )
    
    print("\n[4] Calcolo Permutation Feature Importance...")
    # Usa un subset se il dataset è enorme per risparmiare tempo (10k righe)
    sample_size = min(20000, len(train_data))
    importance_data = train_data.sample(n=sample_size, random_state=42)
    importance = predictor.feature_importance(importance_data)
    
    print("\n    Risultati Permutation Importance:")
    print(importance)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    # Prendi i primi 15 per visibilità
    top_features = importance.head(15)
    plt.barh(top_features.index[::-1], top_features['importance'][::-1], color='mediumpurple')
    plt.title('AutoGluon Permutation Importance (F1-score drop)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('pictures/autogluon_importance.png', dpi=300)
    print("\n[5] Grafico salvato come pictures/autogluon_importance.png")
    
if __name__ == "__main__":
    main()
