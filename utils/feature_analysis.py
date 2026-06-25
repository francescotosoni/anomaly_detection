import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.evaluate_metrics import (
    BASE_DIR, FEATURES, WINDOW_SIZE, STEP_SIZE,
    get_anomaly_files, extract_anomalous_windows
)
from models.LSTM_forecasting import LSTMPredictor

def main():
    print("="*80)
    print("  Feature Importance Analysis: Forecaster Error vs. Random Forest")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("data/scaler.pkl")
    n_features = len(FEATURES)
    
    print(f"\n[1] Caricamento Modello Forecaster e Dati Normali...")
    model = LSTMPredictor(n_features, hidden_dim=32, n_layers=2, dropout=0.2).to(device)
    model.load_state_dict(torch.load("models/lstm_forecasting.pth", weights_only=True))
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    # Load test normal
    X_test_normal = np.load("data/X_test_normal.npy")
    X_test_tensor = torch.tensor(X_test_normal, dtype=torch.float32).to(device)
    
    # Compute baseline normal error per feature
    with torch.no_grad():
        t_input = X_test_tensor[:, :-1, :]
        target = X_test_tensor[:, -1, :]
        recon = model(t_input)
        loss = criterion(recon, target)
        # loss is (batch, features)
        normal_errors_per_feature = loss.mean(dim=0).cpu().numpy()
        
    print("\n[2] Elaborazione file Anomali...")
    anomalous_errors_per_feature = np.zeros(n_features)
    total_anomalous_windows = 0
    
    # For RF
    X_rf = []
    y_rf = []
    
    X_rf.append(X_test_normal[:, -1, :])
    y_rf.append(np.zeros(len(X_test_normal)))
    
    categories = ["change_weight", "drop_object", "force_stop", "outside_box", "push", "scale_speed", "tcp_wrench"]
    for cat in categories:
        files = get_anomaly_files(cat)
        for f in files:
            try:
                df = pd.read_csv(f, delimiter=' ')
                windows, window_labels = extract_anomalous_windows(df, scaler, file_path=f, labeling_mode="tight")
                
                if len(windows) == 0:
                    continue
                    
                # For Forecaster Error
                true_anom_windows = windows[window_labels == 1]
                if len(true_anom_windows) > 0:
                    t_tensor = torch.tensor(true_anom_windows, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        t_input = t_tensor[:, :-1, :]
                        target = t_tensor[:, -1, :]
                        recon = model(t_input)
                        loss = criterion(recon, target)
                        # Accumulate sum of errors
                        anomalous_errors_per_feature += loss.sum(dim=0).cpu().numpy()
                        total_anomalous_windows += len(true_anom_windows)
                        
                # For RF (raw scaled data from windows)
                X_rf.append(windows[:, -1, :])
                y_rf.append(window_labels)
            except Exception as e:
                print(f"Errore nell'elaborazione del file {f}: {e}")
                
    anomalous_errors_per_feature /= max(total_anomalous_windows, 1)
    
    # Calculate Ratio
    ratios = anomalous_errors_per_feature / normal_errors_per_feature
    
    print("\n[3] Metodo 1: Unsupervised Forecaster Error Ratio")
    print(f"    Finestre anomale analizzate: {total_anomalous_windows}")
    # Sort and print
    sorted_idx_1 = np.argsort(ratios)[::-1]
    print("    Top 10 sensori per reattività (Anomalia / Normale):")
    for i in range(10):
        idx = sorted_idx_1[i]
        print(f"      {i+1}. {FEATURES[idx]:<30}: {ratios[idx]:.2f}x")
        
    print("\n[4] Metodo 2: Supervised Random Forest Gini Importance")
    X_rf = np.concatenate(X_rf, axis=0)
    y_rf = np.concatenate(y_rf, axis=0)
    
    print(f"    Training dataset shape: {X_rf.shape}, Anomalies: {np.sum(y_rf)} / {len(y_rf)}")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_rf, y_rf)
    
    importances = rf.feature_importances_
    sorted_idx_2 = np.argsort(importances)[::-1]
    
    print("    Top 10 sensori per Gini Importance (Random Forest):")
    for i in range(10):
        idx = sorted_idx_2[i]
        print(f"      {i+1}. {FEATURES[idx]:<30}: {importances[idx]:.4f}")
        
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1
    features_sorted_1 = [FEATURES[i] for i in sorted_idx_1[:15]]
    ratios_sorted = [ratios[i] for i in sorted_idx_1[:15]]
    axes[0].barh(features_sorted_1[::-1], ratios_sorted[::-1], color='coral')
    axes[0].set_title('Metodo 1: Forecaster Error Ratio')
    axes[0].set_xlabel('Ratio (Errore Anomalia / Errore Normale)')
    
    # Plot 2
    features_sorted_2 = [FEATURES[i] for i in sorted_idx_2[:15]]
    importances_sorted = [importances[i] for i in sorted_idx_2[:15]]
    axes[1].barh(features_sorted_2[::-1], importances_sorted[::-1], color='skyblue')
    axes[1].set_title('Metodo 2: Random Forest Gini Importance')
    axes[1].set_xlabel('Gini Importance')
    
    plt.tight_layout()
    plt.savefig('pictures/feature_importance.png', dpi=300)
    print("\n[5] Grafico salvato come pictures/feature_importance.png")
    
if __name__ == "__main__":
    main()
