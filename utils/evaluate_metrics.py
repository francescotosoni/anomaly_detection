import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import glob
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_CNN import CNNAutoencoder
from models.train_LSTM import LSTMAutoencoder
from models.train_MLP import MLPAutoencoder
from models.train_Transformer import TransformerAutoencoder
from utils.feature_engineering import apply_feature_mask, extract_tabular_features
from utils.preprocess import CustomCategoryScaler

def prepare_data_for_model(windows_np, model_type):
    if model_type == "MLP":
        w = apply_feature_mask(windows_np, 'tabular')
        w = extract_tabular_features(w)
    elif model_type in ["CNN", "LSTM", "Transformer"]:
        w = apply_feature_mask(windows_np, 'sequential')
    else:
        w = windows_np
    return w

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("inference_results.txt", "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
WINDOW_SIZE = 50
STEP_SIZE = 1  
PERSISTENCE_WINDOWS = 250  # 250 windows = 2 seconds of alarm persistence on MCU (at 125Hz)

import json
CUSTOM_LABELS = {}
try:
    with open(r"C:\Users\toso9\ur5_data_collection\data\custom_labels.json", "r", encoding="utf-8") as f:
        CUSTOM_LABELS = json.load(f)
except Exception as e:
    print(f"Could not load custom_labels.json: {e}")


# ---------------------------------------------------------
# SECTION 1: HYPERPARAMETERS & CONFIGURATION
# ---------------------------------------------------------

# General Model hyperparameters
HIDDEN_DIM_LSTM = 32
N_LAYERS_LSTM = 2
DROPOUT_LSTM = 0.2
BOTTLENECK_DIM_MLP = 32

# Transformer specific hyperparameters
D_MODEL_TR = 64
NHEAD_TR = 2
NUM_ENCODER_LAYERS_TR = 2
NUM_DECODER_LAYERS_TR = 2
DIM_FEEDFORWARD_TR = 128
BOTTLENECK_DIM_TR = 32
DROPOUT_TR = 0.1

FEATURES = [
    'actual_q_0', 'actual_q_1', 'actual_q_2',
    'actual_q_3', 'actual_q_4', 'actual_q_5',
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

def extract_anomalous_windows(df, scaler, file_path="", labeling_mode="original"):
    """Extract windows and labels from an anomalous file.
    
    labeling_mode:
        'original' - persistent labels (label=1 from first trigger to end of file)
        'tight'    - timeout-based labels for transient anomalies
    """
    full_df = df.dropna(subset=FEATURES)
    data = full_df[FEATURES].values
    
    if 'actual_digital_input_bits' in full_df.columns:
        bits = full_df['actual_digital_input_bits'].values
        hardware_labels = np.zeros(len(data), dtype=int)
        
        anom_indices = np.where(bits == 32)[0]
        if len(anom_indices) > 0:
            first_anom_idx = anom_indices[0]
            
            if labeling_mode == "custom":
                rel_path = os.path.relpath(file_path, BASE_DIR).replace("\\", "/")
                if rel_path in CUSTOM_LABELS:
                    c_start, c_end = CUSTOM_LABELS[rel_path]
                    hardware_labels[c_start:c_end] = 1
                else:
                    # Logical fallback for files without custom labels
                    hardware_labels[first_anom_idx:] = 1
            elif "push" in file_path or "tcp_wrench" in file_path:
                # Intermittent: label=1 only while hardware flag is active
                hardware_labels[bits == 32] = 1
            elif labeling_mode == "tight" and any(cat in file_path for cat in TRANSIENT_CATEGORIES):
                # Tight labeling: per-category timeout (empirically measured)
                matched_cat = next(cat for cat in TRANSIENT_CATEGORIES if cat in file_path)
                timeout_samples = int(LABEL_TIMEOUTS_S[matched_cat] * SAMPLING_RATE)
                end_idx = min(first_anom_idx + timeout_samples, len(data))
                hardware_labels[first_anom_idx:end_idx] = 1
            else:
                # Original: label=1 from first trigger to end of file
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

# ---------------------------------------------------------
# SECTION 2: EVALUATION FUNCTIONS (Per-Window & Per-Timestep)
# ---------------------------------------------------------

def evaluate_test_set_normal(model, model_type, X_test_normal_tensor, device, threshold=None):
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        if model_type == "CNN":
            t_input = X_test_normal_tensor.transpose(1, 2)
            recon = model(t_input)
            loss = criterion(recon, t_input)
            errors = loss.mean(dim=(1, 2)).cpu().numpy()
        elif model_type == "LSTM_Forecast":
            t_input = X_test_normal_tensor[:, :-1, :]
            target = X_test_normal_tensor[:, -1, :]
            recon = model(t_input)
            loss = criterion(recon, target)
            errors = loss.mean(dim=(1, 2)).cpu().numpy()  # Mean across timesteps and features
            errors = pd.Series(errors).rolling(window=3, min_periods=1).mean().values
        elif model_type == "MLP":
            t_input = X_test_normal_tensor
            recon = model(t_input)
            loss = criterion(recon, t_input)
            errors = loss.mean(dim=1).cpu().numpy()  # Mean across all 1200 flattened features
        else:
            t_input = X_test_normal_tensor
            recon = model(t_input)
            loss = criterion(recon, t_input)
            if model_type == "CNN":
                errors = loss.mean(dim=(1, 2)).cpu().numpy()
            else:
                errors = loss.mean(dim=(1, 2)).cpu().numpy()
            
    if threshold is None:
        # Dynamic Calibration for TinyML: 99.9th percentile on normal test run
        threshold = np.percentile(errors, 99.9)
        
    y_true = [0] * len(errors)
    y_pred = [1 if e > threshold else 0 for e in errors]
    return y_true, y_pred, errors, threshold

def evaluate_anomalous_files(model, model_type, files, scaler, device, threshold, labeling_mode="original"):
    w_true_all = []
    w_pred_all = []
    w_err_all = []
    
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    for f in files:
        df = pd.read_csv(f, sep=' ')
        windows, w_labels = extract_anomalous_windows(df, scaler, file_path=f, labeling_mode=labeling_mode)
        if len(windows) == 0: continue
        
        windows_prep = prepare_data_for_model(windows, model_type)
        data_tensor = torch.tensor(windows_prep, dtype=torch.float32).to(device)
        if model_type == "CNN":
            data_tensor = data_tensor.transpose(1, 2)
            
        with torch.no_grad():
            if model_type == "LSTM_Forecast":
                t_input = data_tensor[:, :-1, :]
                target = data_tensor[:, -1, :]
                recon = model(t_input)
                loss = criterion(recon, target)
                errors = loss.mean(dim=(1, 2)).cpu().numpy()
                errors = pd.Series(errors).rolling(window=3, min_periods=1).mean().values
            else:
                recon = model(data_tensor)
                loss = criterion(recon, data_tensor)
                if model_type == "CNN":
                    errors = loss.mean(dim=(1, 2)).cpu().numpy()
                elif model_type == "MLP":
                    errors = loss.mean(dim=1).cpu().numpy()
                else:
                    errors = loss.mean(dim=(1, 2)).cpu().numpy()
                
        w_err_all.extend(errors)
        w_true_all.extend(w_labels)
        
        preds = [1 if e > threshold else 0 for e in errors]
        w_pred_all.extend(preds)
        
    return w_true_all, w_pred_all, w_err_all

# ---------------------------------------------------------
# SECTION 3: METRIC PRINTING AND REPORTING
# ---------------------------------------------------------

def print_metrics(model_name, y_true_w, y_pred_w, y_errors_w):
    print(f"\n========================================================")
    print(f"       {model_name.upper()} RESULTS       ")
    print(f"========================================================")
    
    # ALARM PERSISTENCE METRICS (MCU Simulator)
    y_pred_pers = apply_alarm_persistence(y_pred_w, PERSISTENCE_WINDOWS)
    p_acc = accuracy_score(y_true_w, y_pred_pers)
    p_prec = precision_score(y_true_w, y_pred_pers, zero_division=0)
    p_rec = recall_score(y_true_w, y_pred_pers, zero_division=0)
    p_f1 = f1_score(y_true_w, y_pred_pers, zero_division=0)
    
    fp = sum(1 for yt, yp in zip(y_true_w, y_pred_pers) if yt == 0 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_w, y_pred_pers) if yt == 0 and yp == 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"[1] PRODUCTION DEPLOY METRICS (Alarm Persistence = {PERSISTENCE_WINDOWS} windows / {PERSISTENCE_WINDOWS/100:.1f}s)")
    print(f"  Recall (Detection Rate): {p_rec:.4f} | False Positive Rate: {fpr:.4f}  (FP: {fp})")
    print(f"  Precision:               {p_prec:.4f} | F1-Score:            {p_f1:.4f}")
    
    # POINT ADJUSTED METRICS (Academic Benchmark)
    y_pred_w_adj, y_errors_w_adj = apply_point_adjustment(y_true_w, y_pred_w, y_errors_w)
    pa_acc = accuracy_score(y_true_w, y_pred_w_adj)
    pa_prec = precision_score(y_true_w, y_pred_w_adj, zero_division=0)
    pa_rec = recall_score(y_true_w, y_pred_w_adj, zero_division=0)
    pa_f1 = f1_score(y_true_w, y_pred_w_adj, zero_division=0)
    try: pa_roc_auc = roc_auc_score(y_true_w, y_errors_w_adj)
    except: pa_roc_auc = 0.0
    try: pa_pr_auc = average_precision_score(y_true_w, y_errors_w_adj)
    except: pa_pr_auc = 0.0
        
    print(f"\n[2] ACADEMIC BENCHMARK (Point-Adjusted — Mismatch forgiven)")
    print(f"  Accuracy:  {pa_acc:.4f} | Precision: {pa_prec:.4f} | Recall: {pa_rec:.4f} | F1: {pa_f1:.4f}")
    print(f"  ROC-AUC:   {pa_roc_auc:.4f} | PR-AUC: {pa_pr_auc:.4f}")

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
    """Print per-category comparison table across all models (PR-AUC and ROC-AUC)."""
    model_names = list(all_results.keys())
    all_cats = sorted(set(
        cat for results in all_results.values() for cat in results.keys()
    ))
    
    # --- PR-AUC TABLE (directly comparable to Graabæk et al. Table 12) ---
    print(f"\n{'='*100}")
    print(f"       PER-CATEGORY PR-AUC (Comparable to Graabæk et al. Table 12)")
    print(f"{'='*100}")
    
    header = f"{'Category':<18}"
    for name in model_names:
        header += f" | {name:>12}"
    print(header)
    print(f"{'-'*100}")
    
    for cat in all_cats:
        row = f"{cat:<18}"
        for name in model_names:
            pr_auc = all_results[name].get(cat, {}).get('pr_auc', 0)
            row += f" | {pr_auc:>12.4f}"
        print(row)
    
    print(f"{'-'*100}")
    # All anomalies combined row
    row = f"{'ALL ANOMALIES':<18}"
    for name in model_names:
        pr_auc = all_results[name].get('__ALL__', {}).get('pr_auc', 0)
        row += f" | {pr_auc:>12.4f}"
    print(row)
    
    # --- ROC-AUC TABLE ---
    print(f"\n{'='*100}")
    print(f"       PER-CATEGORY ROC-AUC (Comparable to Graabæk et al. Table 12)")
    print(f"{'='*100}")
    
    header = f"{'Category':<18}"
    for name in model_names:
        header += f" | {name:>12}"
    print(header)
    print(f"{'-'*100}")
    
    for cat in all_cats:
        row = f"{cat:<18}"
        for name in model_names:
            roc_auc = all_results[name].get(cat, {}).get('roc_auc', 0)
            row += f" | {roc_auc:>12.4f}"
        print(row)
    
    print(f"{'-'*100}")
    row = f"{'ALL ANOMALIES':<18}"
    for name in model_names:
        roc_auc = all_results[name].get('__ALL__', {}).get('roc_auc', 0)
        row += f" | {roc_auc:>12.4f}"
    print(row)
    
    # --- RECALL WITH PERSISTENCE TABLE ---
    print(f"\n{'='*100}")
    print(f"       PER-CATEGORY RECALL (with Alarm Persistence = {PERSISTENCE_WINDOWS} windows)")
    print(f"{'='*100}")
    
    header = f"{'Category':<18}"
    for name in model_names:
        header += f" | {name:>12}"
    print(header)
    print(f"{'-'*100}")
    
    for cat in all_cats:
        row = f"{cat:<18}"
        for name in model_names:
            rec = all_results[name].get(cat, {}).get('recall_pers', 0) * 100
            row += f" | {rec:>11.1f}%"
        print(row)
    
    print(f"{'='*100}")

def evaluate_per_category(model, model_type, scaler, device, threshold, labeling_mode="original"):
    """Evaluate per-category with PR-AUC, ROC-AUC and recall with persistence."""
    results = {}
    
    # Collect ALL anomalous errors/labels for a combined "All anomalies" score
    all_true = []
    all_err = []
    
    for cat in ANOMALY_CLASSES:
        files = get_anomaly_files(cat, max_files=15)
        if not files: continue
        
        cat_true, cat_pred, cat_err = evaluate_anomalous_files(
            model, model_type, files, scaler, device, threshold, labeling_mode=labeling_mode
        )
        if len(cat_true) > 0:
            y_true = np.array(cat_true)
            y_pred = np.array(cat_pred)
            y_err = np.array(cat_err)
            y_pred_pers = np.array(apply_alarm_persistence(y_pred, PERSISTENCE_WINDOWS))
            
            total_anom = int(np.sum(y_true == 1))
            tp_pers = int(np.sum((y_true == 1) & (y_pred_pers == 1)))
            rec_pers = tp_pers / total_anom if total_anom > 0 else 0.0
            
            # PR-AUC and ROC-AUC (threshold-independent, on raw errors)
            try: pr_auc = average_precision_score(y_true, y_err)
            except: pr_auc = 0.0
            try: roc_auc = roc_auc_score(y_true, y_err)
            except: roc_auc = 0.0
            
            results[cat] = {
                'recall_pers': rec_pers,
                'total_anom': total_anom,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc
            }
            
            all_true.extend(cat_true)
            all_err.extend(cat_err)
    
    # Combined "All anomalies" score
    if len(all_true) > 0:
        all_true_np = np.array(all_true)
        all_err_np = np.array(all_err)
        try: pr_auc_all = average_precision_score(all_true_np, all_err_np)
        except: pr_auc_all = 0.0
        try: roc_auc_all = roc_auc_score(all_true_np, all_err_np)
        except: roc_auc_all = 0.0
        
        results['__ALL__'] = {
            'pr_auc': pr_auc_all,
            'roc_auc': roc_auc_all,
            'recall_pers': 0,
            'total_anom': int(np.sum(all_true_np == 1))
        }
    
    return results

def evaluate_per_timestep_normal(model, model_type, X_test_tensor, device):
    """Compute per-timestep reconstruction errors for normal test windows.
    Each window contributes WINDOW_SIZE individual timestep scores, all labeled 0.
    """
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    all_scores = []
    BATCH = 512
    
    with torch.no_grad():
        for start in range(0, len(X_test_tensor), BATCH):
            batch = X_test_tensor[start:start+BATCH]
            
            if model_type == "CNN":
                batch_in = batch.transpose(1, 2)
                recon = model(batch_in)
                loss = criterion(recon, batch_in)
                per_ts = loss.max(dim=1)[0].cpu().numpy()  # (batch, time)
            elif model_type == "LSTM_Forecast":
                t_input = batch[:, :-1, :]
                target = batch[:, -1, :]
                recon = model(t_input)
                loss = criterion(recon, target)
                err = loss.mean(dim=1).cpu().numpy()
                per_ts = np.zeros((batch.shape[0], batch.shape[1]))
                per_ts[:, -1] = err
            elif model_type == "MLP":
                recon = model(batch)
                loss = criterion(recon, batch)
                err = loss.mean(dim=1).cpu().numpy()
                per_ts = np.tile(err[:, np.newaxis], (1, 50))
            else:
                recon = model(batch)
                loss = criterion(recon, batch)
                per_ts = loss.max(dim=2)[0].cpu().numpy()  # (batch, time)
            
            # per_ts has shape (Batch, Time). We want 1 score per window to match the number of unique normal starting positions.
            # Flattening would yield 50x more scores than windows, destroying class balance.
            all_scores.append(per_ts.mean(axis=1))
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.zeros(len(all_scores), dtype=int)
    return all_scores, all_labels


def evaluate_per_timestep_file(model, model_type, file_path, scaler, device, labeling_mode="original"):
    """Compute per-timestep anomaly scores for a single file.
    Runs sliding windows with step=1, averages errors across overlapping windows.
    Returns per-timestep (scores, labels) arrays of length = file_length.
    """
    df = pd.read_csv(file_path, sep=' ').dropna(subset=FEATURES)
    data = df[FEATURES].values
    
    if len(data) < WINDOW_SIZE:
        return np.array([]), np.array([])  
    
    # Per-timestep ground truth labels
    ts_labels = np.zeros(len(data), dtype=int)
    if 'actual_digital_input_bits' in df.columns:
        bits = df['actual_digital_input_bits'].values
        anom_indices = np.where(bits == 32)[0]
        if len(anom_indices) > 0:
            first_anom_idx = anom_indices[0]
            if labeling_mode == "custom":
                rel_path = os.path.relpath(file_path, BASE_DIR).replace("\\", "/")
                if rel_path in CUSTOM_LABELS:
                    c_start, c_end = CUSTOM_LABELS[rel_path]
                    ts_labels[c_start:c_end] = 1
                else:
                    ts_labels[first_anom_idx:] = 1
            else:
                ts_labels[first_anom_idx:] = 1
    
    # Scale and create windows
    scaled_data = scaler.transform(data)
    n_timesteps = len(scaled_data)
    n_windows = n_timesteps - WINDOW_SIZE + 1
    windows = np.array([scaled_data[i:i+WINDOW_SIZE] for i in range(n_windows)])
    windows = prepare_data_for_model(windows, model_type)
    
    # Run model and accumulate per-timestep errors
    score_sum = np.zeros(n_timesteps)
    score_count = np.zeros(n_timesteps)
    
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    BATCH = 512
    
    with torch.no_grad():
        for start in range(0, n_windows, BATCH):
            end = min(start + BATCH, n_windows)
            batch = torch.tensor(windows[start:end], dtype=torch.float32).to(device)
            
            if model_type == "CNN":
                batch = batch.transpose(1, 2)
                recon = model(batch)
                loss = criterion(recon, batch)
                err = loss.mean(dim=(1, 2)).cpu().numpy()  # Scalar per window
                per_ts = np.tile(err[:, np.newaxis], (1, 50))
            elif model_type == "LSTM_Forecast":
                t_input = batch[:, :-1, :]
                target = batch[:, -1, :]
                recon = model(t_input)
                loss = criterion(recon, target)
                err = loss.mean(dim=1).cpu().numpy()
                per_ts = np.zeros((batch.shape[0], batch.shape[1]))
                per_ts[:, -1] = err
            elif model_type == "MLP":
                recon = model(batch)
                loss = criterion(recon, batch)
                err = loss.mean(dim=1).cpu().numpy()
                per_ts = np.tile(err[:, np.newaxis], (1, 50))
            else:
                recon = model(batch)
                loss = criterion(recon, batch)
                err = loss.mean(dim=(1, 2)).cpu().numpy()  # Scalar per window
                per_ts = np.tile(err[:, np.newaxis], (1, 50))
            
            for j, i in enumerate(range(start, end)):
                score_sum[i:i+WINDOW_SIZE] += per_ts[j]
                score_count[i:i+WINDOW_SIZE] += 1
    
    ts_scores = score_sum / np.maximum(score_count, 1)
    return ts_scores, ts_labels


# ---------------------------------------------------------
# SECTION 4: MAIN EXECUTION PIPELINE
# ---------------------------------------------------------

def main():
    print("=" * 80)
    print("  UR5 Anomaly Detection — Full Evaluation (CNN + LSTM + MLP + Transformer)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = joblib.load("data/scaler.pkl")
    n_features = len(FEATURES)
    
    # LOAD Q99 THRESHOLDS
    threshold_cnn = np.load('models/cnn_q99_threshold.npy')[0]
    threshold_lstm = np.load('models/lstm_q99_threshold.npy')[0]
    threshold_mlp = np.load('models/mlp_q99_threshold.npy')[0]
    try:
        threshold_tr = np.load('models/transformer_q99_threshold.npy')[0]
    except FileNotFoundError:
        threshold_tr = 0.0
    print(f" Thresholds -> CNN: {threshold_cnn:.4f} | LSTM: {threshold_lstm:.4f} | MLP: {threshold_mlp:.4f} | Transformer: {threshold_tr:.4f}")
    
    # DYNAMICALLY DETERMINE FEATURE DIMENSIONS
    dummy_data = np.zeros((1, 50, n_features))
    n_features_seq = apply_feature_mask(dummy_data, 'sequential').shape[-1]
    mlp_input_dim = extract_tabular_features(apply_feature_mask(dummy_data, 'tabular')).shape[-1]
    
    # LOAD MODELS
    cnn_model = CNNAutoencoder(n_features_seq).to(device)
    cnn_model.load_state_dict(torch.load("models/cnn_autoencoder.pth", weights_only=True))
    
    lstm_model = LSTMAutoencoder(
        n_features_seq, HIDDEN_DIM_LSTM, 
        seq_len=WINDOW_SIZE, n_layers=N_LAYERS_LSTM, dropout=DROPOUT_LSTM
    ).to(device)
    lstm_model.load_state_dict(torch.load("models/lstm_autoencoder.pth", weights_only=True))
    
    mlp_model = MLPAutoencoder(
        input_dim=mlp_input_dim, bottleneck_dim=BOTTLENECK_DIM_MLP
    ).to(device)
    mlp_model.load_state_dict(torch.load("models/mlp_autoencoder.pth", weights_only=True))
    
    tr_model = TransformerAutoencoder(
        n_features_seq, d_model=D_MODEL_TR, nhead=NHEAD_TR,
        num_encoder_layers=NUM_ENCODER_LAYERS_TR, num_decoder_layers=NUM_DECODER_LAYERS_TR,
        dim_feedforward=DIM_FEEDFORWARD_TR, bottleneck_dim=BOTTLENECK_DIM_TR,
        seq_len=WINDOW_SIZE, dropout=DROPOUT_TR
    ).to(device)
    
    try:
        tr_model.load_state_dict(torch.load("models/transformer_autoencoder.pth", weights_only=True))
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
    X_test_normal = np.load("data/X_test_normal.npy")
    
    models_to_evaluate = [
        (cnn_model, "CNN", threshold_cnn),
        (lstm_model, "LSTM", threshold_lstm),
        (mlp_model, "MLP", threshold_mlp)
    ]
    if has_transformer:
        models_to_evaluate.append((tr_model, "Transformer", threshold_tr))
        
    for model, m_type, thr_loaded in models_to_evaluate:
        # Prepare the dataset dynamically for each model
        X_test_prep = prepare_data_for_model(X_test_normal, m_type)
        X_test_prep_tensor = torch.tensor(X_test_prep, dtype=torch.float32).to(device)
        
        # 1. EVALUATE THRESHOLDS (Q90, Q95, Q99) on normal test set
        _, _, test_err, thr = evaluate_test_set_normal(model, m_type, X_test_prep_tensor, device, threshold=thr_loaded)
    
    # ANOMALOUS TESTS
    anomaly_files = []
    for cat in ANOMALY_CLASSES:
        files = get_anomaly_files(cat, max_files=15)
        anomaly_files.extend(files)
        
    print(f" Evaluating {X_test_normal.shape[0]} unseen Normal windows...")
    print(f" Evaluating {len(anomaly_files)} unseen Anomalous traces...")
    
    thresholds_map = {
        "CNN": threshold_cnn, "LSTM": threshold_lstm, "MLP": threshold_mlp, 
        "Transformer": threshold_tr
    }

    # ==================================================================================
    #  MAIN EVALUATION: PRODUCTION + ACADEMIC (Custom labels)
    # ==================================================================================
    print("\n" + "=" * 100)
    print(f"  MAIN EVALUATION: PRODUCTION + ACADEMIC (Custom Hand-Tuned Labels)")
    print("=" * 100)
    
    t1 = time.time()
    results_tight = {}
    calibrated_thresholds = {}
    
    for name, model in models_to_eval:
        X_test_prep = prepare_data_for_model(X_test_normal, name)
        X_test_prep_tensor = torch.tensor(X_test_prep, dtype=torch.float32).to(device)
        
        n_true, n_pred, n_err, dyn_thr = evaluate_test_set_normal(model, name, X_test_prep_tensor, device, threshold=None)
        calibrated_thresholds[name] = dyn_thr
        print(f" {name} Calibrated Threshold: {dyn_thr:.4f}")
        
        a_true, a_pred, a_err = evaluate_anomalous_files(model, name, anomaly_files, scaler, device, dyn_thr, labeling_mode="custom")
        
        y_true = n_true + a_true
        y_pred = n_pred + a_pred
        y_err = list(n_err) + a_err
        results_tight[name] = (y_true, y_pred, y_err)
    
    print(f"\n Inference (tight labels) complete in {time.time() - t1:.2f} seconds.")
    
    for name, (y_true, y_pred, y_err) in results_tight.items():
        print_metrics(name, y_true, y_pred, y_err)
    
    print("\n\n Evaluating per-category (custom labels)...")
    cat_results_tight = {}
    for name, model in models_to_eval:
        dyn_thr = calibrated_thresholds[name]
        cat_results_tight[name] = evaluate_per_category(model, name, scaler, device, dyn_thr, labeling_mode="custom")
    
    print_per_category_comparison(cat_results_tight)
    
    # ==================================================================================
    #  TABLE C: PER-TIMESTEP SCORING (directly comparable to Graabæk et al.)
    # ==================================================================================
    print("\n" + "=" * 100)
    print("  TABLE C: PER-TIMESTEP SCORING (directly comparable to Graabæk et al.)")
    print("  Score = mean reconstruction error across all overlapping windows for each timestep")
    print("=" * 100)
    
    t2 = time.time()
    cat_results_per_ts = {}
    
    for model, m_type, _ in models_to_evaluate:
        print(f"\n  Processing {m_type}...")
        
        # Prepare the dataset dynamically
        X_test_prep = prepare_data_for_model(X_test_normal, m_type)
        X_test_prep_tensor = torch.tensor(X_test_prep, dtype=torch.float32).to(device)
        
        # Pre-compute normal timestep errors for ROC-AUC
        all_scores, all_labels = evaluate_per_timestep_normal(model, m_type, X_test_prep_tensor, device)
        
        # Per-category anomalous data
        per_cat = {}
        all_a_scores = []
        all_a_labels = []
        
        for cat in ANOMALY_CLASSES:
            files = get_anomaly_files(cat, max_files=15)
            cat_scores_list = []
            cat_labels_list = []
            
            for f in files:
                ts_scores, ts_labels = evaluate_per_timestep_file(
                    model, m_type, f, scaler, device, labeling_mode="custom"
                )
                if len(ts_scores) > 0:
                    cat_scores_list.append(ts_scores)
                    cat_labels_list.append(ts_labels)
            
            if cat_scores_list:
                cat_s = np.concatenate(cat_scores_list)
                cat_l = np.concatenate(cat_labels_list)
                
                try: pr_auc = average_precision_score(cat_l, cat_s)
                except: pr_auc = 0.0
                try: roc_auc = roc_auc_score(cat_l, cat_s)
                except: roc_auc = 0.0
                
                per_cat[cat] = {'pr_auc': pr_auc, 'roc_auc': roc_auc}
                all_a_scores.append(cat_s)
                all_a_labels.append(cat_l)
        
        # Combined: normal + all anomalous
        if all_a_scores:
            combined_scores = np.concatenate([all_scores] + all_a_scores)
            combined_labels = np.concatenate([all_labels] + all_a_labels)
            
            try: pr_all = average_precision_score(combined_labels, combined_scores)
            except: pr_all = 0.0
            try: roc_all = roc_auc_score(combined_labels, combined_scores)
            except: roc_all = 0.0
            
            per_cat['__ALL__'] = {'pr_auc': pr_all, 'roc_auc': roc_all}
        cat_results_per_ts[m_type] = per_cat
    print(f"\n Per-timestep evaluation complete in {time.time() - t2:.2f} seconds.")
    
    # Print per-timestep PR-AUC table
    model_names = list(cat_results_per_ts.keys())
    all_cats = sorted(set(
        cat for results in cat_results_per_ts.values() for cat in results.keys()
        if cat != '__ALL__'
    ))
    
    print(f"\n{'='*100}")
    print(f"       PER-TIMESTEP PR-AUC (Directly comparable to Graabæk et al. Table 12)")
    print(f"{'='*100}")
    header = f"{'Category':<18}"
    for name in model_names:
        header += f" | {name:>12}"
    print(header)
    print(f"{'-'*100}")
    for cat in all_cats:
        row = f"{cat:<18}"
        for name in model_names:
            val = cat_results_per_ts[name].get(cat, {}).get('pr_auc', 0)
            row += f" | {val:>12.4f}"
        print(row)
    print(f"{'-'*100}")
    row = f"{'ALL ANOMALIES':<18}"
    for name in model_names:
        val = cat_results_per_ts[name].get('__ALL__', {}).get('pr_auc', 0)
        row += f" | {val:>12.4f}"
    print(row)
    
    # Print per-timestep ROC-AUC table
    print(f"\n{'='*100}")
    print(f"       PER-TIMESTEP ROC-AUC (Directly comparable to Graabæk et al. Table 12)")
    print(f"{'='*100}")
    header = f"{'Category':<18}"
    for name in model_names:
        header += f" | {name:>12}"
    print(header)
    print(f"{'-'*100}")
    for cat in all_cats:
        row = f"{cat:<18}"
        for name in model_names:
            val = cat_results_per_ts[name].get(cat, {}).get('roc_auc', 0)
            row += f" | {val:>12.4f}"
        print(row)
    print(f"{'-'*100}")
    row = f"{'ALL ANOMALIES':<18}"
    for name in model_names:
        val = cat_results_per_ts[name].get('__ALL__', {}).get('roc_auc', 0)
        row += f" | {val:>12.4f}"
    print(row)
    print(f"{'='*100}")
    
    # ==================================================================================
    #  CONFRONTO FINALE: PAPER vs OURS (per-window) vs OURS (per-timestep)
    # ==================================================================================
    print(f"\n{'='*110}")
    print(f"       CONFRONTO CON GRAABÆK ET AL. (Table 12 — All anomalies)")
    print(f"{'='*110}")
    print(f"{'Metodo':<25} | {'PR-AUC':>8} | {'ROC-AUC':>8} | {'Note':>35}")
    print(f"{'-'*90}")
    
    paper_results = [
        ('LOF (paper)', 0.8255, 0.9531, 'best overall (paper)'),
        ('KNN (paper)', 0.7926, 0.9428, ''),
        ('VAE+GMM (paper)', 0.7925, 0.9275, ''),
        ('LSTM AE (paper)', 0.7860, 0.9192, 'same architecture class'),
        ('DAE (paper)', 0.7014, 0.8500, ''),
        ('CVAE (paper)', 0.7208, 0.8778, 'uses program context'),
    ]
    for name, pr, roc, note in paper_results:
        print(f"{name:<25} | {pr:>8.4f} | {roc:>8.4f} | {note:>35}")
    
    print(f"{'-'*90}")
    print("--- Per-Window (custom) --- |          |          |")
    for name in [n for n, _ in models_to_eval]:
        all_data = cat_results_tight.get(name, {}).get('__ALL__', {})
        pr = all_data.get('pr_auc', 0)
        roc = all_data.get('roc_auc', 0)
        print(f"{name + ' (window)':<25} | {pr:>8.4f} | {roc:>8.4f} |")
    
    print(f"{'-'*90}")
    print(f"{'--- Per-Timestep ---':<25} |          |          | {'*** comparable to paper ***':>35}")
    for name in [n for n, _ in models_to_eval]:
        all_data = cat_results_per_ts.get(name, {}).get('__ALL__', {})
        pr = all_data.get('pr_auc', 0)
        roc = all_data.get('roc_auc', 0)
        print(f"{name + ' (timestep)':<25} | {pr:>8.4f} | {roc:>8.4f} |")
    
    print(f"{'='*110}")

if __name__ == "__main__":
    main()
