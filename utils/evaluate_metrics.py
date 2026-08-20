import argparse
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
    """Tee stdout to a file. Installed from main(), not at import time, so that
    other modules can import the helpers here without their output being
    hijacked."""
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- CONFIGURATION ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT, "data", "data", "stochastic_PnP")
WINDOW_SIZE = 50
STEP_SIZE = 1
DEBOUNCE_WINDOWS = 5       # 5 windows (40ms at 125Hz) over threshold to trigger alarm
PERSISTENCE_WINDOWS = 250  # 250 windows = 2 seconds of alarm persistence on MCU (at 125Hz)

# Ground truth: one uint8 mask per run, produced by utils/merge_labels.py.
# Per-sample rather than [start, end], because the published masks are not
# contiguous - on change_weight the label switches off while the robot stands
# still, since a wrong payload is unobservable when nothing moves.
MERGED_LABELS = {}

def load_merged_labels(path):
    global MERGED_LABELS
    try:
        MERGED_LABELS = {k: v.astype(np.uint8) for k, v in np.load(path).items()}
        print(f" Loaded {len(MERGED_LABELS)} merged label masks from {os.path.basename(path)}")
    except Exception as e:
        raise SystemExit(f"Could not load {path}: {e}\nRun: python utils/merge_labels.py")

def label_mask(file_path, n_samples):
    """Merged ground-truth mask for a run, or None when unavailable."""
    rel = os.path.relpath(file_path, BASE_DIR).replace("\\", "/")
    m = MERGED_LABELS.get(rel)
    if m is None:
        return None
    if len(m) >= n_samples:
        return m[:n_samples]
    out = np.zeros(n_samples, dtype=np.uint8)
    out[:len(m)] = m
    return out


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

# The eight folders map one-to-one onto the eight anomalies of Graabaek et al.;
# force_stop is the paper's "Raised object".
#
# long_wait is excluded by construction, not because it is hard. The anomaly is a
# 12-20 s wait in place of the nominal 4-10 s, while a window is 50 samples at
# 100 Hz, i.e. 0.5 s. A window inside a 20 s pause and one inside a 6 s pause hold
# the same thing - a stationary robot - so no score computed on that window can
# separate them. Detecting it needs the contextual Time_in_node feature the paper
# uses and the deployment target does not have.
ALL_ANOMALY_CLASSES = [
    "push", "tcp_wrench", "drop_object", "change_weight",
    "outside_box", "scale_speed", "force_stop", "long_wait"
]
ANOMALY_CLASSES = [c for c in ALL_ANOMALY_CLASSES if c != "long_wait"]

def get_anomaly_files(category, max_files=None):
    folder = os.path.join(BASE_DIR, category)
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "test", "*.csv"))
    if max_files:
        files = files[:max_files]
    return files

def apply_debouncing(y_pred, debounce_windows=DEBOUNCE_WINDOWS):
    """Anti-noise filter: requires N consecutive windows to exceed threshold."""
    y_pred_deb = np.zeros_like(y_pred)
    count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            count += 1
            if count >= debounce_windows:
                y_pred_deb[i] = 1
        else:
            count = 0
    return y_pred_deb.tolist()

def calculate_segment_metrics(y_true, y_pred):
    """Event-Based Evaluation: counts True Positives and False Positives by segments."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Identify true anomaly segments
    true_diff = np.diff(np.concatenate(([0], y_true, [0])))
    true_starts = np.where(true_diff == 1)[0]
    true_ends = np.where(true_diff == -1)[0]
    
    # Identify predicted segments
    pred_diff = np.diff(np.concatenate(([0], y_pred, [0])))
    pred_starts = np.where(pred_diff == 1)[0]
    pred_ends = np.where(pred_diff == -1)[0]
    
    tp_events = 0
    fn_events = 0
    fp_events = 0
    
    # Check True Positives and False Negatives
    for s, e in zip(true_starts, true_ends):
        if np.any(y_pred[s:e] == 1):
            tp_events += 1
        else:
            fn_events += 1
            
    # Check False Positives (predicted segments that don't overlap with ANY true segment)
    for ps, pe in zip(pred_starts, pred_ends):
        overlap = False
        for ts, te in zip(true_starts, true_ends):
            if not (pe <= ts or ps >= te):
                overlap = True
                break
        if not overlap:
            fp_events += 1
            
    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) > 0 else 0.0
    recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1, tp_events, fp_events, fn_events

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

def extract_anomalous_windows(df, scaler, file_path="", labeling_mode="merged"):
    """Extract windows and labels from an anomalous file.
    
    labeling_mode:
        'merged'   - the per-sample mask from utils/merge_labels.py (default)
        'original' - raw button channel, label=1 from first trigger to end of file
    """
    full_df = df.dropna(subset=FEATURES)
    data = full_df[FEATURES].values

    hardware_labels = np.zeros(len(data), dtype=int)

    if labeling_mode == "merged":
        mask = label_mask(file_path, len(data))
        if mask is not None:
            hardware_labels = mask.astype(int)
        elif 'actual_digital_input_bits' in full_df.columns:
            # No merged mask for this run: fall back to the raw channel rather
            # than silently scoring it as entirely nominal.
            bits = full_df['actual_digital_input_bits'].values
            anom = np.where(bits == 32)[0]
            if len(anom):
                hardware_labels[anom[0]:] = 1
    elif 'actual_digital_input_bits' in full_df.columns:
        bits = full_df['actual_digital_input_bits'].values
        anom_indices = np.where(bits == 32)[0]
        if len(anom_indices) > 0:
            if "push" in file_path or "tcp_wrench" in file_path:
                # Intermittent: label=1 only while the hardware flag is active
                hardware_labels[bits == 32] = 1
            else:
                hardware_labels[anom_indices[0]:] = 1

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
        threshold = 0.0 # Will be dynamically calibrated later
        
    y_true = [0] * len(errors)
    y_pred = [1 if e > threshold else 0 for e in errors]
    return y_true, y_pred, errors, threshold

def evaluate_anomalous_files(model, model_type, files, scaler, device, threshold, labeling_mode="merged"):
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
    y_pred_deb = apply_debouncing(y_pred_w, DEBOUNCE_WINDOWS)
    y_pred_pers = apply_alarm_persistence(y_pred_deb, PERSISTENCE_WINDOWS)
    
    # Point-by-point stats (just for debug/reference)
    p_prec = precision_score(y_true_w, y_pred_pers, zero_division=0)
    p_rec = recall_score(y_true_w, y_pred_pers, zero_division=0)
    p_f1 = f1_score(y_true_w, y_pred_pers, zero_division=0)
    
    # Event-Based Evaluation
    seg_prec, seg_rec, seg_f1, tp, fp, fn = calculate_segment_metrics(y_true_w, y_pred_pers)
    
    print(f"[1] PRODUCTION DEPLOY METRICS (Debounce={DEBOUNCE_WINDOWS}, Persistence={PERSISTENCE_WINDOWS})")
    print(f"  --- Event-Based (Segments) ---")
    print(f"  Precision: {seg_prec:.4f} | Recall: {seg_rec:.4f} | F1-Score: {seg_f1:.4f}")
    print(f"  Events -> TP: {tp} | FP: {fp} | FN: {fn}")
    print(f"  --- Point-by-Point (Legacy) ---")
    print(f"  Precision: {p_prec:.4f} | Recall: {p_rec:.4f} | F1-Score: {p_f1:.4f}")

def find_optimal_threshold(y_true, y_errors, name="Model", percentiles=np.arange(90, 100, 0.5)):
    """Sweeps thresholds and returns the one that maximizes Segment-Based F1-Score."""
    print(f"\n--- {name}: Optimizing Threshold ---")
    
    # Extract normal errors to generate threshold candidates
    normal_errors = [e for t, e in zip(y_true, y_errors) if t == 0]
    if len(normal_errors) == 0: 
        return np.mean(y_errors)
        
    best_f1 = -1
    best_thr = 0
    best_metrics = {}
    
    for q in percentiles:
        thr = np.percentile(normal_errors, q)
        y_pred = [1 if e > thr else 0 for e in y_errors]
        
        y_pred_deb = apply_debouncing(y_pred, DEBOUNCE_WINDOWS)
        y_pred_pers = apply_alarm_persistence(y_pred_deb, PERSISTENCE_WINDOWS)
        
        seg_prec, seg_rec, seg_f1, tp, fp, fn = calculate_segment_metrics(y_true, y_pred_pers)
        
        if seg_f1 > best_f1:
            best_f1 = seg_f1
            best_thr = thr
            best_metrics = {'q': q, 'prec': seg_prec, 'rec': seg_rec, 'f1': seg_f1, 'fp': fp}
            
    print(f"  Optimal Threshold found at q{best_metrics['q']}: {best_thr:.4f}")
    print(f"  Expected Event F1: {best_metrics['f1']:.4f} (FP events: {best_metrics['fp']})")
    
    return best_thr

def print_per_category_comparison(all_results):
    """Print per-category comparison table across all models (PR-AUC and ROC-AUC)."""
    model_names = list(all_results.keys())
    all_cats = sorted(set(
        cat for results in all_results.values() for cat in results.keys()
        if cat != '__ALL__'
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

def evaluate_per_category(model, model_type, scaler, device, threshold, n_true=None, n_err=None, labeling_mode="merged"):
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
            y_pred_deb = apply_debouncing(y_pred, DEBOUNCE_WINDOWS)
            y_pred_pers = np.array(apply_alarm_persistence(y_pred_deb, PERSISTENCE_WINDOWS))
            
            seg_prec, seg_rec, seg_f1, tp, fp, fn = calculate_segment_metrics(y_true, y_pred_pers)
            total_anom = tp + fn
            
            # PR-AUC and ROC-AUC (threshold-independent, on raw errors)
            try: pr_auc = average_precision_score(y_true, y_err)
            except: pr_auc = 0.0
            try: roc_auc = roc_auc_score(y_true, y_err)
            except: roc_auc = 0.0
            
            results[cat] = {
                'recall_pers': seg_rec,
                'tp': tp,
                'total_anom': total_anom,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc
            }
            
            all_true.extend(cat_true)
            all_err.extend(cat_err)
    
    # Combined "All anomalies" score
    if len(all_true) > 0:
        if n_true is not None and n_err is not None:
            all_true_np = np.array(list(n_true) + all_true)
            all_err_np = np.array(list(n_err) + all_err)
        else:
            all_true_np = np.array(all_true)
            all_err_np = np.array(all_err)
            
        try: pr_auc_all = average_precision_score(all_true_np, all_err_np)
        except: pr_auc_all = 0.0
        try: roc_auc_all = roc_auc_score(all_true_np, all_err_np)
        except: roc_auc_all = 0.0
        
        total_tp = sum(results[cat].get('tp', 0) for cat in results)
        total_anom_events = sum(results[cat].get('total_anom', 0) for cat in results)
        recall_all = total_tp / total_anom_events if total_anom_events > 0 else 0.0
        
        results['__ALL__'] = {
            'pr_auc': pr_auc_all,
            'roc_auc': roc_auc_all,
            'recall_pers': recall_all,
            'total_anom': total_anom_events
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
                err = loss.mean(dim=(1, 2)).cpu().numpy()
            elif model_type == "LSTM_Forecast":
                t_input = batch[:, :-1, :]
                target = batch[:, -1, :]
                recon = model(t_input)
                loss = criterion(recon, target)
                err = loss.mean(dim=1).cpu().numpy()
            elif model_type == "MLP":
                recon = model(batch)
                loss = criterion(recon, batch)
                err = loss.mean(dim=1).cpu().numpy()
            else:
                recon = model(batch)
                loss = criterion(recon, batch)
                err = loss.mean(dim=(1, 2)).cpu().numpy()
            
            all_scores.append(err)
    
    all_scores = np.concatenate(all_scores)
    all_labels = np.zeros(len(all_scores), dtype=int)
    return all_scores, all_labels


def evaluate_per_timestep_file(model, model_type, file_path, scaler, device, labeling_mode="merged"):
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
    mask = label_mask(file_path, len(data)) if labeling_mode == "merged" else None
    if mask is not None:
        ts_labels = mask.astype(int)
    elif 'actual_digital_input_bits' in df.columns:
        bits = df['actual_digital_input_bits'].values
        anom_indices = np.where(bits == 32)[0]
        if len(anom_indices) > 0:
            ts_labels[anom_indices[0]:] = 1


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

def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Full evaluation against the merged ground truth")
    ap.add_argument("--suffix", default="",
                    help="training run to evaluate: picks data/scaler<suffix>.pkl, "
                         "data/X_test_normal<suffix>.npy and models/*<suffix>.pth")
    ap.add_argument("--models", nargs="+", default=None,
                    choices=["CNN", "LSTM", "MLP", "Transformer"],
                    help="models to evaluate (default: every one found on disk)")
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "merged_labels.npz"))
    ap.add_argument("--categories", nargs="+", default=None, choices=ALL_ANOMALY_CLASSES,
                    help="anomaly categories to evaluate (default: all but long_wait)")
    ap.add_argument("--out", default=None, help="report file (default: inference_results<suffix>.txt)")
    return ap


def main():
    args = build_arg_parser().parse_args()
    global ANOMALY_CLASSES
    if args.categories:
        ANOMALY_CLASSES = list(args.categories)
    sfx = args.suffix
    sys.stdout = Logger(args.out or os.path.join(ROOT, f"inference_results{sfx}.txt"))

    print("=" * 80)
    print(f"  UR5 Anomaly Detection - Full Evaluation{f' [{sfx.lstrip(chr(95))}]' if sfx else ''}")
    print("=" * 80)

    load_merged_labels(args.labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_path = os.path.join(ROOT, "data", f"scaler{sfx}.pkl")
    scaler = joblib.load(scaler_path)
    print(f" Scaler: {os.path.basename(scaler_path)}")
    n_features = len(FEATURES)

    # DYNAMICALLY DETERMINE FEATURE DIMENSIONS
    dummy_data = np.zeros((1, 50, n_features))
    n_features_seq = apply_feature_mask(dummy_data, 'sequential').shape[-1]
    mlp_input_dim = extract_tabular_features(apply_feature_mask(dummy_data, 'tabular')).shape[-1]

    # Model constructors, instantiated only for the checkpoints that exist for
    # this training run. Mixing suffixes would pair a model with a scaler it was
    # not trained under, so a missing checkpoint is skipped rather than
    # substituted with the unsuffixed one.
    builders = {
        "CNN": lambda: CNNAutoencoder(n_features_seq),
        "LSTM": lambda: LSTMAutoencoder(n_features_seq, HIDDEN_DIM_LSTM, seq_len=WINDOW_SIZE,
                                        n_layers=N_LAYERS_LSTM, dropout=DROPOUT_LSTM),
        "MLP": lambda: MLPAutoencoder(input_dim=mlp_input_dim, bottleneck_dim=BOTTLENECK_DIM_MLP),
        "Transformer": lambda: TransformerAutoencoder(
            n_features_seq, d_model=D_MODEL_TR, nhead=NHEAD_TR,
            num_encoder_layers=NUM_ENCODER_LAYERS_TR, num_decoder_layers=NUM_DECODER_LAYERS_TR,
            dim_feedforward=DIM_FEEDFORWARD_TR, bottleneck_dim=BOTTLENECK_DIM_TR,
            seq_len=WINDOW_SIZE, dropout=DROPOUT_TR),
    }
    checkpoints = {"CNN": "cnn_autoencoder", "LSTM": "lstm_autoencoder",
                   "MLP": "mlp_autoencoder", "Transformer": "transformer_autoencoder"}

    wanted = args.models or list(builders)
    models_to_eval = []
    for name in wanted:
        path = os.path.join(ROOT, "models", f"{checkpoints[name]}{sfx}.pth")
        if not os.path.exists(path):
            print(f" [!] {name}: {os.path.basename(path)} not found, skipped")
            continue
        model = builders[name]().to(device)
        model.load_state_dict(torch.load(path, weights_only=True))
        models_to_eval.append((name, model))
    if not models_to_eval:
        raise SystemExit("no model to evaluate")

    for name, model in models_to_eval:
        params = sum(p.numel() for p in model.parameters())
        size_kb = params * 4 / 1024
        print(f" {name}: {params:,} params ({size_kb:.1f} KB float32)")
    
    # PURE NORMAL TEST
    print("\n Loading held-out nominal windows...")
    X_test_normal = np.load(os.path.join(ROOT, "data", f"X_test_normal{sfx}.npy"), mmap_mode="r")
    
    # ANOMALOUS TESTS
    anomaly_files = []
    for cat in ANOMALY_CLASSES:
        files = get_anomaly_files(cat, max_files=15)
        anomaly_files.extend(files)
        
    print(f" Evaluating {X_test_normal.shape[0]} unseen Normal windows...")
    print(f" Evaluating {len(anomaly_files)} unseen Anomalous traces...")
    # ==================================================================================
    print("\n" + "=" * 100)
    print(f"  MAIN EVALUATION: PRODUCTION DEPLOY (Debounce + Segment-based)")
    print("=" * 100)
    
    t1 = time.time()
    results_tight = {}
    calibrated_thresholds = {}
    normal_data_cache = {}
    
    for name, model in models_to_eval:
        X_test_prep = prepare_data_for_model(X_test_normal, name)
        X_test_prep_tensor = torch.tensor(X_test_prep, dtype=torch.float32).to(device)
        
        n_true, _, n_err, _ = evaluate_test_set_normal(model, name, X_test_prep_tensor, device, threshold=None)
        normal_data_cache[name] = (n_true, n_err)
        
        # Evaluate anomalous files with threshold 0.0 to get errors
        a_true, _, a_err = evaluate_anomalous_files(model, name, anomaly_files, scaler, device, 0.0, labeling_mode="merged")
        
        y_true = n_true + a_true
        y_err = list(n_err) + a_err
        
        # 1. Optimize threshold
        dyn_thr = find_optimal_threshold(y_true, y_err, name=name)
        calibrated_thresholds[name] = dyn_thr
        
        # 2. Final prediction
        y_pred = [1 if e > dyn_thr else 0 for e in y_err]
        results_tight[name] = (y_true, y_pred, y_err)
    
    print(f"\n Inference (tight labels) complete in {time.time() - t1:.2f} seconds.")
    
    for name, (y_true, y_pred, y_err) in results_tight.items():
        print_metrics(name, y_true, y_pred, y_err)
    
    print("\n\n Evaluating per-category (merged labels)...")
    cat_results_tight = {}
    for name, model in models_to_eval:
        dyn_thr = calibrated_thresholds[name]
        n_t, n_e = normal_data_cache[name]
        cat_results_tight[name] = evaluate_per_category(model, name, scaler, device, dyn_thr, n_true=n_t, n_err=n_e, labeling_mode="merged")
    
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
    
    models_to_evaluate = [(m, n, calibrated_thresholds[n]) for n, m in models_to_eval]
    
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
                    model, m_type, f, scaler, device, labeling_mode="merged"
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
    #  FINAL COMPARISON: PAPER vs OURS (per-window) vs OURS (per-timestep)
    # ==================================================================================
    print(f"\n{'='*110}")
    print(f"       COMPARISON WITH GRAABAEK ET AL. (Table 12 - All anomalies)")
    print(f"{'='*110}")
    print(f"{'Method':<25} | {'PR-AUC':>8} | {'ROC-AUC':>8} | {'Notes':>35}")
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
