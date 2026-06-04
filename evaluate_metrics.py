import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import glob
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from train_CNN import CNNAutoencoder
from train_LSTM import LSTMAutoencoder
from train_MLP import MLPAutoencoder
from train_Transformer import TransformerAutoencoder

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
WINDOW_SIZE = 50
STEP_SIZE = 1  
PERSISTENCE_WINDOWS = 200  # 200 finestre = 2 secondi di allarme su MCU (a 100Hz)

# Modelli parametri
HIDDEN_DIM_LSTM = 32
N_LAYERS_LSTM = 2
DROPOUT_LSTM = 0.2
BOTTLENECK_DIM_MLP = 32

# Transformer parametri
D_MODEL_TR = 64
NHEAD_TR = 2
NUM_ENCODER_LAYERS_TR = 2
NUM_DECODER_LAYERS_TR = 2
DIM_FEEDFORWARD_TR = 128
BOTTLENECK_DIM_TR = 32
DROPOUT_TR = 0.1

FEATURES = [
    'actual_current_0', 'actual_current_1', 'actual_current_2', 
    'actual_current_3', 'actual_current_4', 'actual_current_5',
    'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 
    'actual_qd_3', 'actual_qd_4', 'actual_qd_5',
    'actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2',
    'actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5'
]

ANOMALY_CLASSES = [
    "push", "tcp_wrench", "drop_object", "change_weight", 
    "force_stop", "outside_box", "scale_speed"
]

def get_anomaly_files(category, max_files=None):
    folder = os.path.join(BASE_DIR, category)
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "test", "*.csv"))
    if max_files:
        files = files[:max_files]
    return files

def apply_point_adjustment(y_true, y_pred, y_errors):
    """Academic Benchmark adjustment."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_errors = np.array(y_errors)
    
    y_pred_adj = np.copy(y_pred)
    y_errors_adj = np.copy(y_errors)
    
    diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for s, e in zip(starts, ends):
        if np.any(y_pred[s:e] == 1):
            y_pred_adj[s:e] = 1
            y_errors_adj[s:e] = np.max(y_errors[s:e])
            
    return y_pred_adj.tolist(), y_errors_adj.tolist()

def apply_alarm_persistence(y_pred, persistence=PERSISTENCE_WINDOWS):
    """Simulates MCU behavior: if alarm triggers, it stays active for `persistence` windows."""
    y_pred_pers = np.copy(y_pred)
    active_until = -1
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            active_until = i + persistence
        if i <= active_until:
            y_pred_pers[i] = 1
    return y_pred_pers.tolist()

def extract_anomalous_windows(df, scaler, file_path=""):
    full_df = df.dropna(subset=FEATURES)
    data = full_df[FEATURES].values
    
    if 'actual_digital_input_bits' in full_df.columns:
        bits = full_df['actual_digital_input_bits'].values
        hardware_labels = np.zeros(len(data), dtype=int)
        
        anom_indices = np.where(bits == 32)[0]
        if len(anom_indices) > 0:
            first_anom_idx = anom_indices[0]
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

def evaluate_test_set_normal(model, model_type, X_test_normal_tensor, device, threshold):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        if model_type == "CNN":
            t_input = X_test_normal_tensor.transpose(1, 2)
            recon = model(t_input)
            loss = criterion(recon, t_input)
            errors = loss.mean(dim=2).max(dim=1)[0].cpu().numpy()
        else:
            t_input = X_test_normal_tensor
            recon = model(t_input)
            loss = criterion(recon, t_input)
            errors = loss.mean(dim=1).max(dim=1)[0].cpu().numpy()
            
    y_true = [0] * len(errors)
    y_pred = [1 if e > threshold else 0 for e in errors]
    return y_true, y_pred, errors

def evaluate_anomalous_files(model, model_type, files, scaler, device, threshold):
    w_true_all = []
    w_pred_all = []
    w_err_all = []
    
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    for f in files:
        df = pd.read_csv(f, sep=' ')
        windows, w_labels = extract_anomalous_windows(df, scaler, file_path=f)
        if len(windows) == 0: continue
        
        data_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
        if model_type == "CNN":
            data_tensor = data_tensor.transpose(1, 2)
            
        with torch.no_grad():
            recon = model(data_tensor)
            loss = criterion(recon, data_tensor)
            if model_type == "CNN":
                errors = loss.mean(dim=2).max(dim=1)[0].cpu().numpy()
            else:
                errors = loss.mean(dim=1).max(dim=1)[0].cpu().numpy()
                
        w_err_all.extend(errors)
        w_true_all.extend(w_labels)
        
        # We compute predictions per file so we can apply file-level persistence later if needed
        # but for simplicity we apply it globally or just pass raw arrays
        preds = [1 if e > threshold else 0 for e in errors]
        w_pred_all.extend(preds)
        
    return w_true_all, w_pred_all, w_err_all

def print_metrics(model_name, y_true_w, y_pred_w, y_errors_w):
    print(f"\n========================================================")
    print(f"       {model_name.upper()} RESULTS       ")
    print(f"========================================================")
    
    # RAW METRICS
    w_acc = accuracy_score(y_true_w, y_pred_w)
    w_prec = precision_score(y_true_w, y_pred_w, zero_division=0)
    w_rec = recall_score(y_true_w, y_pred_w, zero_division=0)
    w_f1 = f1_score(y_true_w, y_pred_w, zero_division=0)
    try: w_roc_auc = roc_auc_score(y_true_w, y_errors_w)
    except: w_roc_auc = 0.0
        
    print(f"[1] RAW METRICS (Strict window-by-window, flawed labeling)")
    print(f"  Accuracy:  {w_acc:.4f} | Precision: {w_prec:.4f} | Recall: {w_rec:.4f} | F1: {w_f1:.4f} | AUC: {w_roc_auc:.4f}")
    
    # ALARM PERSISTENCE METRICS (MCU Simulator)
    y_pred_pers = apply_alarm_persistence(y_pred_w, PERSISTENCE_WINDOWS)
    p_acc = accuracy_score(y_true_w, y_pred_pers)
    p_prec = precision_score(y_true_w, y_pred_pers, zero_division=0)
    p_rec = recall_score(y_true_w, y_pred_pers, zero_division=0)
    p_f1 = f1_score(y_true_w, y_pred_pers, zero_division=0)
    print(f"\n[2] MCU DEPLOY METRICS (Alarm Persistence = {PERSISTENCE_WINDOWS} windows / {PERSISTENCE_WINDOWS/100:.1f}s)")
    print(f"  Accuracy:  {p_acc:.4f} | Precision: {p_prec:.4f} | Recall: {p_rec:.4f} | F1: {p_f1:.4f}")
    
    # POINT ADJUSTED METRICS (Academic Benchmark)
    y_pred_w_adj, y_errors_w_adj = apply_point_adjustment(y_true_w, y_pred_w, y_errors_w)
    pa_acc = accuracy_score(y_true_w, y_pred_w_adj)
    pa_prec = precision_score(y_true_w, y_pred_w_adj, zero_division=0)
    pa_rec = recall_score(y_true_w, y_pred_w_adj, zero_division=0)
    pa_f1 = f1_score(y_true_w, y_pred_w_adj, zero_division=0)
    try: pa_roc_auc = roc_auc_score(y_true_w, y_errors_w_adj)
    except: pa_roc_auc = 0.0
        
    print(f"\n[3] POINT-ADJUSTED (Academic benchmark - Mismatch forgiven)")
    print(f"  Accuracy:  {pa_acc:.4f} | Precision: {pa_prec:.4f} | Recall: {pa_rec:.4f} | F1: {pa_f1:.4f} | AUC: {pa_roc_auc:.4f}")

def evaluate_threshold_sweep(y_true, y_errors, name="Model"):
    """Evaluates different thresholds based on test set percentiles."""
    print(f"\n--- {name} Threshold Sweep (on Test Normal) ---")
    
    # Extract just the normal test errors to calculate percentiles
    normal_errors = [e for t, e in zip(y_true, y_errors) if t == 0]
    
    if len(normal_errors) == 0: return
    
    thresholds = {
        "q90": np.percentile(normal_errors, 90),
        "q95": np.percentile(normal_errors, 95),
        "q99": np.percentile(normal_errors, 99)
    }
    
    print(f"{'Threshold':<10} | {'Value':<8} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Persist. F1':<11}")
    print("-" * 75)
    for q, thr in thresholds.items():
        y_pred = [1 if e > thr else 0 for e in y_errors]
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        y_pred_pers = apply_alarm_persistence(y_pred, PERSISTENCE_WINDOWS)
        f1_pers = f1_score(y_true, y_pred_pers, zero_division=0)
        
        print(f"{q:<10} | {thr:<8.3f} | {prec:<9.4f} | {rec:<9.4f} | {f1:<9.4f} | {f1_pers:<11.4f}")

def print_per_category_comparison(all_results):
    """Print per-category comparison table across all models."""
    model_names = list(all_results.keys())
    all_cats = sorted(set(
        cat for results in all_results.values() for cat in results.keys()
    ))
    
    print(f"\n{'='*95}")
    print(f"       PER-CATEGORY ANOMALY DETECTION RATE (Recall with Alarm Persistence)")
    print(f"{'='*95}")
    
    header = f"{'Category':<18}"
    for name in model_names:
        header += f" | {name:>10}"
    header += f" | {'Anom. Win.':>10}"
    print(header)
    print(f"{'-'*95}")
    
    for cat in all_cats:
        row = f"{cat:<18}"
        for name in model_names:
            r = all_results[name].get(cat, {}).get('recall_pers', 0) * 100
            row += f" | {r:>9.1f}%"
        n_anom = 0
        for name in model_names:
            n = all_results[name].get(cat, {}).get('total_anom', 0)
            if n > 0:
                n_anom = n
                break
        row += f" | {n_anom:>10}"
        print(row)
    
    print(f"{'='*95}")

def evaluate_per_category(model, model_type, scaler, device, threshold):
    results = {}
    for cat in ANOMALY_CLASSES:
        files = get_anomaly_files(cat, max_files=15)
        if not files: continue
        
        cat_true, cat_pred, cat_err = evaluate_anomalous_files(
            model, model_type, files, scaler, device, threshold
        )
        if len(cat_true) > 0:
            y_true = np.array(cat_true)
            y_pred = np.array(cat_pred)
            y_pred_pers = np.array(apply_alarm_persistence(y_pred, PERSISTENCE_WINDOWS))
            
            total_anom = int(np.sum(y_true == 1))
            tp_pers = int(np.sum((y_true == 1) & (y_pred_pers == 1)))
            rec_pers = tp_pers / total_anom if total_anom > 0 else 0.0
            
            results[cat] = {
                'recall_pers': rec_pers,
                'total_anom': total_anom
            }
    return results

def main():
    print("=" * 80)
    print("  UR5 Anomaly Detection — Full Evaluation (CNN + LSTM + MLP + Transformer)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("scaler.pkl")
    n_features = len(FEATURES)
    
    # LOAD Q99 THRESHOLDS
    threshold_cnn = np.load('cnn_q99_threshold.npy')[0]
    threshold_lstm = np.load('lstm_q99_threshold.npy')[0]
    threshold_mlp = np.load('mlp_q99_threshold.npy')[0]
    try:
        threshold_tr = np.load('transformer_q99_threshold.npy')[0]
    except FileNotFoundError:
        threshold_tr = 0.0  # fallback if not trained yet
        
    print(f" Thresholds -> CNN: {threshold_cnn:.4f} | LSTM: {threshold_lstm:.4f} | MLP: {threshold_mlp:.4f} | Transformer: {threshold_tr:.4f}")
    
    # LOAD MODELS
    cnn_model = CNNAutoencoder(n_features).to(device)
    cnn_model.load_state_dict(torch.load("cnn_autoencoder.pth", weights_only=True))
    
    lstm_model = LSTMAutoencoder(
        n_features, HIDDEN_DIM_LSTM, 
        seq_len=WINDOW_SIZE, n_layers=N_LAYERS_LSTM, dropout=DROPOUT_LSTM
    ).to(device)
    lstm_model.load_state_dict(torch.load("lstm_autoencoder.pth", weights_only=True))
    
    mlp_model = MLPAutoencoder(
        WINDOW_SIZE, n_features, bottleneck_dim=BOTTLENECK_DIM_MLP
    ).to(device)
    mlp_model.load_state_dict(torch.load("mlp_autoencoder.pth", weights_only=True))
    
    tr_model = TransformerAutoencoder(
        n_features, d_model=D_MODEL_TR, nhead=NHEAD_TR,
        num_encoder_layers=NUM_ENCODER_LAYERS_TR, num_decoder_layers=NUM_DECODER_LAYERS_TR,
        dim_feedforward=DIM_FEEDFORWARD_TR, bottleneck_dim=BOTTLENECK_DIM_TR,
        seq_len=WINDOW_SIZE, dropout=DROPOUT_TR
    ).to(device)
    
    try:
        tr_model.load_state_dict(torch.load("transformer_autoencoder.pth", weights_only=True))
        has_transformer = True
    except FileNotFoundError:
        print(" [!] Transformer model not found. Skipping its evaluation.")
        has_transformer = False
    
    # Print model sizes
    models_to_eval = [("CNN", cnn_model), ("LSTM", lstm_model), ("MLP", mlp_model)]
    if has_transformer: models_to_eval.append(("Transformer", tr_model))
    
    for name, model in models_to_eval:
        params = sum(p.numel() for p in model.parameters())
        size_kb = params * 4 / 1024
        print(f" {name}: {params:,} params ({size_kb:.1f} KB float32)")
    
    # PURE NORMAL TEST
    print("\n Loading Pure Isolated X_test_normal.npy...")
    X_test_normal = np.load("X_test_normal.npy")
    X_test_tensor = torch.tensor(X_test_normal, dtype=torch.float32).to(device)
    
    # ANOMALOUS TESTS
    anomaly_files = []
    for cat in ANOMALY_CLASSES:
        files = get_anomaly_files(cat, max_files=15)
        anomaly_files.extend(files)
        
    print(f" Evaluating {X_test_normal.shape[0]} unseen Normal windows...")
    print(f" Evaluating {len(anomaly_files)} unseen Anomalous traces...")

    t0 = time.time()
    
    results_all = {}
    
    # RUN EVALUATIONS
    for name, model in models_to_eval:
        thr = {"CNN": threshold_cnn, "LSTM": threshold_lstm, "MLP": threshold_mlp, "Transformer": threshold_tr}[name]
        
        n_true, n_pred, n_err = evaluate_test_set_normal(model, name, X_test_tensor, device, thr)
        a_true, a_pred, a_err = evaluate_anomalous_files(model, name, anomaly_files, scaler, device, thr)
        
        y_true = n_true + a_true
        y_pred = n_pred + a_pred
        y_err = list(n_err) + a_err
        
        results_all[name] = (y_true, y_pred, y_err)

    print(f"\n Inference complete in {time.time() - t0:.2f} seconds.")

    # PRINT METRICS
    for name, (y_true, y_pred, y_err) in results_all.items():
        print_metrics(name, y_true, y_pred, y_err)
        evaluate_threshold_sweep(y_true, y_err, name=name)
    
    # PER-CATEGORY BREAKDOWN
    print("\n\n Evaluating per-category detection rates...")
    cat_results = {}
    for name, model in models_to_eval:
        thr = {"CNN": threshold_cnn, "LSTM": threshold_lstm, "MLP": threshold_mlp, "Transformer": threshold_tr}[name]
        cat_results[name] = evaluate_per_category(model, name, scaler, device, thr)
        
    print_per_category_comparison(cat_results)

if __name__ == "__main__":
    main()
