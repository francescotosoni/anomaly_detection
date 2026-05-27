import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import glob
import os
import time
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from train_CNN import CNNAutoencoder
from train_LSTM import LSTMAutoencoder

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
WINDOW_SIZE = 50
STEP_SIZE = 1  # <-- RETURNED TO 1: Frame resolution as per Graabæk
HIDDEN_DIM_LSTM = 32

FEATURES = [
    'actual_current_0', 'actual_current_1', 'actual_current_2', 
    'actual_current_3', 'actual_current_4', 'actual_current_5',
    'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 
    'actual_qd_3', 'actual_qd_4', 'actual_qd_5',
    'actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2',
    'actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5'
]

# Healthy and Anomalous categories
NORMAL_CLASS = "normal"
ANOMALY_CLASSES = [
    "push", "tcp_wrench", "drop_object", "change_weight", 
    "force_stop", "outside_box", "scale_speed"
]

def get_files(category, max_files=None):
    folder = os.path.join(BASE_DIR, category)
    # Recursively find CSVs
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "test", "*.csv"))
    if max_files:
        files = files[:max_files]
    return files

def extract_windows(df, scaler, file_path=""):
    full_df = df.dropna(subset=FEATURES)
    data = full_df[FEATURES].values
    
    if 'actual_digital_input_bits' in full_df.columns:
        bits = full_df['actual_digital_input_bits'].values
        hardware_labels = np.zeros(len(data), dtype=int)
        
        anom_indices = np.where(bits == 32)[0]
        if len(anom_indices) > 0:
            first_anom_idx = anom_indices[0]
            
            # Apply anomaly extension based on type (as described in Graabæk paper)
            # - Push: only during contact (bits == 32).
            # - Drop / Scale speed / Long wait etc: from trigger point onwards.
            if "push" in file_path or "tcp_wrench" in file_path:
                hardware_labels[bits == 32] = 1
            else:
                hardware_labels[first_anom_idx:] = 1
    else:
        hardware_labels = np.zeros(len(data), dtype=int)

    scaled_data = scaler.transform(data)
    
    windows = []
    window_labels = []
    for i in range(0, len(scaled_data) - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(scaled_data[i : i + WINDOW_SIZE])
        window_labels.append(1 if 1 in hardware_labels[i : i + WINDOW_SIZE] else 0)
        
    return np.array(windows), np.array(window_labels)

def evaluate_files(model, model_type, files, scaler, device, threshold, is_anomaly=False):
    file_predictions = []
    file_true_labels = []
    
    window_predictions_all = []
    window_true_labels_all = []
    window_errors_all = []

    criterion = nn.MSELoss(reduction='none')
    
    model.eval()
    
    for f in files:
        df = pd.read_csv(f, sep=' ')
        windows, w_labels = extract_windows(df, scaler, file_path=f)
        if len(windows) == 0:
            continue
            
        data_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
        if model_type == "CNN":
            data_tensor = data_tensor.transpose(1, 2)
            
        with torch.no_grad():
            reconstruction = model(data_tensor)
            loss = criterion(reconstruction, data_tensor)
            
            if model_type == "CNN":
                errors = loss.mean(dim=2).max(dim=1)[0].cpu().numpy()
            else:
                errors = loss.mean(dim=1).max(dim=1)[0].cpu().numpy()
            
        PERSISTENCE = 1 # No debouncing filter to challenge strict Window-Level
        is_pred_file_anomaly = 0
        consecutive = 0
        
        w_preds = []
        # Graabxk paper: calculate window performance where PR curves aren't heavily debounced
        # We'll just collect errors for raw PR curve
        for err in errors:
            if err > threshold:
                consecutive += 1
                w_pred = 1 if consecutive >= PERSISTENCE else 0
            else:
                consecutive = 0
                w_pred = 0
                
            w_preds.append(w_pred)    
            if w_pred == 1:
                is_pred_file_anomaly = 1
                
        file_predictions.append(is_pred_file_anomaly)
        file_true_labels.append(1 if is_anomaly else 0)
        
        window_true_labels_all.extend(w_labels)
        window_predictions_all.extend(w_preds)
        window_errors_all.extend(errors)
    
    return file_true_labels, file_predictions, window_true_labels_all, window_predictions_all, window_errors_all

def apply_point_adjustment(y_true, y_pred, y_errors):
    """
    Point-Adjusted (PA) Metric for Time-Series Anomaly Detection.
    If a continuous anomalous segment (y_true == 1) contains AT LEAST ONE frame predicted as anomalous (y_pred == 1),
    then the ENTIRE segment is considered DETECTED (false negatives within the segment are cancelled out).
    Adapted also for errors (ROC-AUC PA) by "spreading" the maximum recorded error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_errors = np.array(y_errors)
    
    y_pred_adj = np.copy(y_pred)
    y_errors_adj = np.copy(y_errors)
    
    # Identify start and end of anomalous segments (1)
    diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for s, e in zip(starts, ends):
        if np.any(y_pred[s:e] == 1):
            # If alarm triggers, the whole segment is considered correct and known
            y_pred_adj[s:e] = 1
            # To not penalize ROC-AUC (which depends on errors, not boolean threshold)
            # We give the entire segment a 'prob/error' equal to the max observed in that segment
            max_err_in_segment = np.max(y_errors[s:e])
            y_errors_adj[s:e] = max_err_in_segment
            
    return y_pred_adj.tolist(), y_errors_adj.tolist()

def print_metrics(model_name, y_true_f, y_pred_f, y_true_w, y_pred_w, y_errors_w):
    # --- File-Level Metrics ---
    f_acc = accuracy_score(y_true_f, y_pred_f)
    f_prec = precision_score(y_true_f, y_pred_f, zero_division=0)
    f_rec = recall_score(y_true_f, y_pred_f, zero_division=0)
    f_f1 = f1_score(y_true_f, y_pred_f, zero_division=0)
    
    y_pred_w_adj, y_errors_w_adj = apply_point_adjustment(y_true_w, y_pred_w, y_errors_w)
    
    # --- Window-Level Metrics ---
    w_acc = accuracy_score(y_true_w, y_pred_w_adj)
    w_prec = precision_score(y_true_w, y_pred_w_adj, zero_division=0)
    w_rec = recall_score(y_true_w, y_pred_w_adj, zero_division=0)
    w_f1 = f1_score(y_true_w, y_pred_w_adj, zero_division=0)
    
    # ROC-AUC calculation as done by Graabæk et al.
    try:
        w_roc_auc = roc_auc_score(y_true_w, y_errors_w_adj)
        # Also calculate PR-AUC (desired on unbalanced datasets like WL)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_w, y_errors_w_adj)
        w_pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        w_roc_auc = 0.0
        w_pr_auc = 0.0
    
    print(f"\n=====================================")
    print(f"       {model_name.upper()} RESULTS       ")
    print(f"=====================================")
    print(f"▶ FILE-LEVEL (Whole Trace)")
    print(f"  Accuracy:  {f_acc:.4f}")
    print(f"  Precision: {f_prec:.4f}")
    print(f"  Recall:    {f_rec:.4f}")
    print(f"  F1-Score:  {f_f1:.4f}")
    
    print(f"\n▶ WINDOW-LEVEL (Graabæk's Metric)")
    print(f"  Accuracy:  {w_acc:.4f}")
    print(f"  Precision: {w_prec:.4f}")
    print(f"  Recall:    {w_rec:.4f}")
    print(f"  F1-Score:  {w_f1:.4f} (Point-Adjusted)")
    print(f"  ROC-AUC:   {w_roc_auc:.4f} (Point-Adjusted)")
    print(f"  PR-AUC:    {w_pr_auc:.4f} (Point-Adjusted)")
    
    print("\nWindow-Level Confusion Matrix (Point-Adjusted):")
    print(classification_report(y_true_w, y_pred_w_adj, target_names=["Normal", "Anomaly"]))

def main():
    print("Preparing File-Level Metric Evaluation...")
    scaler = joblib.load("scaler.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(FEATURES)
    
    # Load CNN
    cnn_model = CNNAutoencoder(n_features).to(device)
    cnn_model.load_state_dict(torch.load("cnn_autoencoder.pth", weights_only=True))
    
    # Load LSTM
    lstm_model = LSTMAutoencoder(n_features, HIDDEN_DIM_LSTM).to(device)
    lstm_model.load_state_dict(torch.load("lstm_autoencoder.pth", weights_only=True))
    
    # Thresholds: 95th Percentile of the Maximum Error of the ENTIRE HEALTHY FILE
    # Calculated on X_normal to discard physiological vibrations
    threshold_cnn = 6.63
    threshold_lstm = 26.02
    
    # Collect files
    # Limit to 50 healthy files to keep the class balance fair
    normal_files = get_files(NORMAL_CLASS, max_files=100)
    print(f"Evaluating {len(normal_files)} Normal traces...")
    
    anomaly_files = []
    # Collect up to 20 files per anomaly class to have roughly 140 anonalies
    for cat in ANOMALY_CLASSES:
        files = get_files(cat, max_files=15)
        anomaly_files.extend(files)
        
    print(f"Evaluating {len(anomaly_files)} Anomalous traces across {len(ANOMALY_CLASSES)} categories...")

    # Evaluate Normal
    t0 = time.time()
    res_cnn_n = evaluate_files(cnn_model, "CNN", normal_files, scaler, device, threshold_cnn, is_anomaly=False)
    res_lstm_n = evaluate_files(lstm_model, "LSTM", normal_files, scaler, device, threshold_lstm, is_anomaly=False)
    
    # Evaluate Anomalous
    res_cnn_a = evaluate_files(cnn_model, "CNN", anomaly_files, scaler, device, threshold_cnn, is_anomaly=True)
    res_lstm_a = evaluate_files(lstm_model, "LSTM", anomaly_files, scaler, device, threshold_lstm, is_anomaly=True)
    
    print(f"Inference completed in {time.time() - t0:.2f} seconds.")

    # Combine Results per CNN
    CNN_y_true_f = res_cnn_n[0] + res_cnn_a[0]
    CNN_y_pred_f = res_cnn_n[1] + res_cnn_a[1]
    CNN_y_true_w = res_cnn_n[2] + res_cnn_a[2]
    CNN_y_pred_w = res_cnn_n[3] + res_cnn_a[3]
    CNN_y_errors_w = res_cnn_n[4] + res_cnn_a[4]
    
    # Combine Results per LSTM
    LSTM_y_true_f = res_lstm_n[0] + res_lstm_a[0]
    LSTM_y_pred_f = res_lstm_n[1] + res_lstm_a[1]
    LSTM_y_true_w = res_lstm_n[2] + res_lstm_a[2]
    LSTM_y_pred_w = res_lstm_n[3] + res_lstm_a[3]
    LSTM_y_errors_w = res_lstm_n[4] + res_lstm_a[4]
    
    # Print reports
    print_metrics("CNN 1D Autoencoder", CNN_y_true_f, CNN_y_pred_f, CNN_y_true_w, CNN_y_pred_w, CNN_y_errors_w)
    print_metrics("LSTM Autoencoder", LSTM_y_true_f, LSTM_y_pred_f, LSTM_y_true_w, LSTM_y_pred_w, LSTM_y_errors_w)

if __name__ == "__main__":
    main()