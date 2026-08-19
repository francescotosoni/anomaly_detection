"""
Recalibration (domain adaptation) of the anomaly detection models on the real
conditions of the laboratory.

Methodology (as rigorous as models/train_*.py, but on lab data):

  * Only NORMAL runs in training. Anomalous files (push_*, weight_*) are excluded
    by construction: an autoencoder that saw the anomalies would learn to
    reconstruct them, destroying its ability to detect them.
  * PER-RUN split, not per-window. With STEP_SIZE=1 the windows of one run
    overlap by ~98%: a random split over windows would put near-clones in both
    train and validation, and the q99 threshold would come out optimistic.
  * Scaler fitted ONLY on training runs (validation stays held out).
  * Early stopping + LR scheduler + best-model selection on validation loss.
  * q99 threshold computed on VALIDATION windows and THROUGH the wrapper
    ONNX wrapper, hence on the same scale as the score the bridge sees at runtime.
    (The original scripts use a different aggregation -- mean over time, MAX over
    features -- which yields thresholds ~8x off the ONNX scale.)

Reuses the real ARCHITECTURES (models/train_*.py) and the WRAPPERS from
utils/export_to_onnx.py: the recalibrated models stay faithful to the benchmark
ones and comparable to each other. Output in models/ICE/ (models/onnx/ untouched).

Usage:
  python lab_retrain.py --data data/lab/task1                  # every model
  python lab_retrain.py --data data/lab/task1 --model cnn mlp
  python lab_retrain.py --data data/lab/task1 data/lab/task2   # modello combinato
  python lab_retrain.py --data data/lab/task1 --stride 5       # faster
"""

import sys
import os
import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn

# Console UTF-8 (i moduli importati stampano emoji: evita UnicodeEncodeError su cp1252)
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# --- Path setup: rende importabili models/ e utils/ ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Architetture reali
from models.train_CNN import CNNAutoencoder, CNNUpAutoencoder
from models.train_LSTM import LSTMAutoencoder
from models.train_MLP import MLPAutoencoder
from models.train_Transformer import TransformerAutoencoder
# Wrapper ONNX (input unificato (batch,50,24) -> output (batch,1) = anomaly score)
from utils.export_to_onnx import (
    CNNAnomalyScoreWrapper,
    AnomalyScoreWrapper,
    MLPAnomalyScoreWrapper,
    export_model_to_onnx,
)
from utils.feature_engineering import (
    apply_feature_mask,
    extract_tabular_features,
    TAB_FEATURES_TO_DROP,
)
# I/O condiviso: 24 feature canoniche + lettura CSV con separatore auto-rilevato
from utils.lab_io import read_telemetry, features_array, resolve_csv_paths, run_label, run_label

# ==========================================
# CONFIGURAZIONE
# ==========================================
# Output folder: set by main() via --out. Every experiment goes into its own
# sottocartella dedicata (es. models/ICE/task1, models/ICE/task2, models/ICE/combined)
# otherwise one run overwrites the previous and cross-task comparison is impossible.
DEFAULT_OUTPUT_DIR = os.path.join("models", "ICE")
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
SCALER_OUTPUT = os.path.join(OUTPUT_DIR, "scaler.json")
MANIFEST_OUTPUT = os.path.join(OUTPUT_DIR, "manifest.json")


def set_output_dir(path):
    """Set the output folder and the derived paths."""
    global OUTPUT_DIR, SCALER_OUTPUT, MANIFEST_OUTPUT
    OUTPUT_DIR = path
    SCALER_OUTPUT = os.path.join(path, "scaler.json")
    MANIFEST_OUTPUT = os.path.join(path, "manifest.json")
    os.makedirs(path, exist_ok=True)

WINDOW_SIZE = 50
N_FEATURES = 24
BATCH_SIZE = 128
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 10
DEFAULT_VAL_RUNS = 3      # run tenute fuori dal training (per soglia e early stopping)
DEFAULT_SEED = 42
LEARNING_RATE = 1e-3

# Pattern for NORMAL files: anomalies must not enter training
NORMAL_GLOB = 'run*.csv'
ANOMALY_PREFIXES = ('push', 'weight', 'drop', 'wrench', 'force_stop', 'outside', 'scale')

# Architecture hyperparameters (in sync with export_to_onnx.py)
BOTTLENECK_MLP = 32
HIDDEN_LSTM, NLAYERS_LSTM, DROPOUT_LSTM = 32, 2, 0.2
DMODEL_TR, NHEAD_TR, ENC_TR, DEC_TR, FF_TR, BOTTLENECK_TR, DROPOUT_TR = 64, 2, 2, 2, 128, 32, 0.1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# SCALER (identico a preprocess.CustomCategoryScaler)
# ==========================================
CATEGORIES = {'q': (0, 6), 'current': (6, 12), 'qd': (12, 18), 'tcp': (18, 24)}


def calculate_scaler(values):
    """mean/std per feature category (shared across the 6 columns of each group)."""
    params = {}
    for cat, (start, stop) in CATEGORIES.items():
        cat_data = values[:, start:stop]
        std = float(np.std(cat_data))
        params[cat] = {'mean': float(np.mean(cat_data)), 'std': std if std != 0 else 1.0,
                       'slice': [start, stop]}
    return params


def write_scaler_json(params):
    os.makedirs(os.path.dirname(SCALER_OUTPUT), exist_ok=True)
    with open(SCALER_OUTPUT, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"   [OK] Scaler -> {SCALER_OUTPUT} (bridge.py loads it from here)")


def _scale(values, params):
    """Apply category scaling to an (M, 24) array."""
    scaled = values.copy()
    for cat, p in params.items():
        start, stop = p['slice']
        scaled[:, start:stop] = (scaled[:, start:stop] - p['mean']) / p['std']
    return scaled


def _windows(scaled, stride=1):
    """Overlapping windows from a SINGLE continuous file -> (n, 50, 24) float32."""
    n = len(scaled) - WINDOW_SIZE + 1
    if n <= 0:
        return np.empty((0, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
    return np.stack([scaled[i:i + WINDOW_SIZE] for i in range(0, n, stride)]).astype(np.float32)


# ==========================================
# FILE SELECTION AND PER-RUN SPLIT
# ==========================================
def collect_normal_files(paths):
    """
    Resolve the paths keeping ONLY the normal runs.
    Directories are expanded with NORMAL_GLOB; files passed explicitly that carry
    an anomaly prefix are rejected with a warning.
    """
    expanded = []
    for p in paths:
        if os.path.isdir(p):
            found = resolve_csv_paths([os.path.join(p, NORMAL_GLOB)])
            all_csv = resolve_csv_paths([p])
            skipped = [os.path.basename(f) for f in all_csv if f not in found]
            print(f"   {p}: {len(found)} run normali ({NORMAL_GLOB})"
                  + (f", esclusi {len(skipped)} file non-normali: {', '.join(skipped)}" if skipped else ""))
            expanded += found
        else:
            expanded += resolve_csv_paths([p])

    keep = []
    for f in expanded:
        if os.path.basename(f).lower().startswith(ANOMALY_PREFIXES):
            print(f"   [!] {os.path.basename(f)} looks like an ANOMALY file -> excluded from training")
            continue
        keep.append(f)
    return keep


def load_raw_per_file(files):
    """Load each CSV -> (name, (M,24) array), discarding short or incomplete files."""
    per_file = []
    for f in files:
        try:
            vals = features_array(read_telemetry(f))
        except Exception as e:
            print(f"   [!] {f}: {e} -> saltato")
            continue
        if len(vals) < WINDOW_SIZE:
            print(f"   [!] {f}: solo {len(vals)} campioni (<{WINDOW_SIZE}) -> saltato")
            continue
        per_file.append((f, vals))
    return per_file


def split_runs(per_file, n_val, seed):
    """
    Split the runs into train/validation. Random split with a fixed seed
    (reproducible): simply taking the last runs would introduce a systematic bias,
    because the robot warms up during a session and the currents drift.
    """
    if len(per_file) < 2:
        raise RuntimeError(
            f"at least 2 runs are needed for the per-run split (found {len(per_file)}). "
            "Record more normal runs with collect.py.")
    n_val = max(1, min(n_val, len(per_file) - 1))
    order = np.random.RandomState(seed).permutation(len(per_file))
    val_idx = set(order[:n_val].tolist())
    train = [pf for i, pf in enumerate(per_file) if i not in val_idx]
    val = [pf for i, pf in enumerate(per_file) if i in val_idx]
    return train, val


# ==========================================
# TRAINING (early stopping + scheduler + best model)
# ==========================================
def train_autoencoder(model, train_input, val_input, epochs, patience,
                      grad_clip=None, weight_decay=0.0):
    """
    train_input/val_input: tensors already in the format the model expects.
    Returns the model with the weights of the BEST epoch on validation loss.
    """
    model = model.to(DEVICE)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_input), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_input), batch_size=512, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=15, T_mult=2)
    crit = nn.MSELoss()

    best_val, best_state, best_epoch, stale = float('inf'), None, 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            tot += loss.item() * xb.size(0)
        train_loss = tot / len(train_input)
        sched.step()

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                vtot += crit(model(xb), xb).item() * xb.size(0)
        val_loss = vtot / len(val_input)

        improved = val_loss < best_val
        if improved:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1

        if epoch % 5 == 0 or epoch == 1 or improved:
            mark = ' *' if improved else ''
            print(f"     epoch {epoch:3d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}{mark}")

        if stale >= patience:
            print(f"     early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"     modello migliore: epoca {best_epoch}, val loss {best_val:.6f}")
    # If the minimum is close to the last epoch without early stopping, the epoch
    # cap interrupted a training run still improving: the model is UNDERTRAINED and
    # comparing it with a converged one would be unfair.
    converged = stale >= patience
    if not converged and best_epoch > epochs - patience:
        print(f"     [!] NOT CONVERGED: no early stopping and minimum at epoch {best_epoch}/{epochs}. "
              f"Re-run with a larger --epochs (e.g. {epochs * 2}).")
    return model.eval(), best_val, best_epoch, converged


def _scores(wrapped, X):
    """Score di anomalia via wrapper (stessa scala dell'ONNX esportato)."""
    wrapped = wrapped.to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            xb = torch.tensor(X[i:i + 512]).to(DEVICE)
            out.append(wrapped(xb).cpu().numpy().ravel())
    return np.concatenate(out)


def threshold_from_validation(wrapped, X_train, X_val, name):
    """
    q99 threshold on the VALIDATION windows (held out) through the wrapper.
    Also reports the in-sample q99, to make visible the optimism one would get
    by computing the threshold on the training data.
    """
    s_val = _scores(wrapped, X_val)
    s_tr = _scores(wrapped, X_train)
    q99_val = float(np.quantile(s_val, 0.99))
    q99_tr = float(np.quantile(s_tr, 0.99))
    np.save(os.path.join(OUTPUT_DIR, f"{name}_q99_threshold.npy"), np.array([q99_val]))
    ratio = q99_val / max(q99_tr, 1e-12)
    print(f"     score val:   media={s_val.mean():.5f}  q99={q99_val:.5f}  <- SOGLIA (held-out)")
    print(f"     score train: media={s_tr.mean():.5f}  q99={q99_tr:.5f}  (in-sample)")
    # A ratio well above 1 means the model reconstructs the training data better
    # than new data (overfitting): the in-sample threshold would have been too low
    # and produced false alarms in deployment. Close to 1 = good generalisation.
    print(f"     held-out/in-sample ratio = {ratio:.2f}x"
          + ("  [wide gap: the in-sample threshold would have caused false alarms]" if ratio > 1.3 else ""))
    return dict(q99=q99_val, q99_in_sample=q99_tr,
                val_mean=float(s_val.mean()), train_mean=float(s_tr.mean()))


# ==========================================
# PER-MODEL RECALIBRATION
# ==========================================
def _finish(wrapped, name, X_train, X_val, trained):
    """trained = tupla (model, best_val, best_epoch, converged) da train_autoencoder."""
    _, best_val, best_epoch, converged = trained
    export_model_to_onnx(wrapped, name, torch.randn(1, WINDOW_SIZE, N_FEATURES), OUTPUT_DIR)
    info = threshold_from_validation(wrapped, X_train, X_val, name)
    info.update(val_loss=best_val, best_epoch=best_epoch, converged=bool(converged))
    return info


def recalibrate_cnn(X_train, X_val, args):
    print("\n>> CNN")
    trained = train_autoencoder(
        CNNAutoencoder(N_FEATURES),
        torch.tensor(X_train).transpose(1, 2),   # (N,24,50) channels-first
        torch.tensor(X_val).transpose(1, 2),
        args.epochs, args.patience, grad_clip=1.0, weight_decay=1e-4)
    return _finish(CNNAnomalyScoreWrapper(trained[0]), "cnn_autoencoder", X_train, X_val, trained)


def recalibrate_cnn_up(X_train, X_val, args):
    """CNN with an Upsample+Conv decoder instead of ConvTranspose: same sizes,
    but every operator is accelerated by ESP-NN on the ESP32-S3."""
    print("\n>> CNN-UP (decoder Upsample + Conv, senza ConvTranspose)")
    trained = train_autoencoder(
        CNNUpAutoencoder(N_FEATURES),
        torch.tensor(X_train).transpose(1, 2),   # (N,24,50) channels-first
        torch.tensor(X_val).transpose(1, 2),
        args.epochs, args.patience, grad_clip=1.0, weight_decay=1e-4)
    return _finish(CNNAnomalyScoreWrapper(trained[0]), "cnn_up_autoencoder", X_train, X_val, trained)


def recalibrate_mlp(X_train, X_val, args):
    print("\n>> MLP (feature engineering: 19 feature x 5 statistiche = 95)")
    f_tr = extract_tabular_features(apply_feature_mask(X_train, model_type='tabular')).astype(np.float32)
    f_va = extract_tabular_features(apply_feature_mask(X_val, model_type='tabular')).astype(np.float32)
    trained = train_autoencoder(
        MLPAutoencoder(input_dim=f_tr.shape[1], bottleneck_dim=BOTTLENECK_MLP),
        torch.tensor(f_tr), torch.tensor(f_va),
        args.epochs, args.patience, weight_decay=1e-5)
    return _finish(MLPAnomalyScoreWrapper(trained[0], TAB_FEATURES_TO_DROP),
                   "mlp_autoencoder", X_train, X_val, trained)


def recalibrate_lstm(X_train, X_val, args):
    print("\n>> LSTM")
    trained = train_autoencoder(
        LSTMAutoencoder(N_FEATURES, HIDDEN_LSTM, seq_len=WINDOW_SIZE,
                        n_layers=NLAYERS_LSTM, dropout=DROPOUT_LSTM),
        torch.tensor(X_train), torch.tensor(X_val), args.epochs, args.patience)
    return _finish(AnomalyScoreWrapper(trained[0]), "lstm_autoencoder", X_train, X_val, trained)


def recalibrate_transformer(X_train, X_val, args):
    print("\n>> Transformer")
    trained = train_autoencoder(
        TransformerAutoencoder(N_FEATURES, d_model=DMODEL_TR, nhead=NHEAD_TR,
                               num_encoder_layers=ENC_TR, num_decoder_layers=DEC_TR,
                               dim_feedforward=FF_TR, bottleneck_dim=BOTTLENECK_TR,
                               seq_len=WINDOW_SIZE, dropout=DROPOUT_TR),
        torch.tensor(X_train), torch.tensor(X_val), args.epochs, args.patience)
    return _finish(AnomalyScoreWrapper(trained[0]), "transformer_autoencoder", X_train, X_val, trained)


RECALIBRATORS = {
    'cnn': recalibrate_cnn,
    'cnn_up': recalibrate_cnn_up,
    'mlp': recalibrate_mlp,
    'lstm': recalibrate_lstm,
    'transformer': recalibrate_transformer,
}


# ==========================================
# MAIN
# ==========================================
def main():
    ap = argparse.ArgumentParser(description="Model recalibration on real laboratory data")
    ap.add_argument('--model', nargs='+', default=['all'],
                    choices=['all', 'cnn', 'cnn_up', 'mlp', 'lstm', 'transformer'],
                    help="Which models to recalibrate (default: all)")
    ap.add_argument('--data', nargs='+', required=True,
                    help="Folders/files/globs with the NORMAL runs. Folders are expanded "
                         f"with '{NORMAL_GLOB}': anomalous files are excluded automatically.")
    ap.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    ap.add_argument('--patience', type=int, default=DEFAULT_PATIENCE,
                    help="Epochs without improvement before early stopping")
    ap.add_argument('--val-runs', type=int, default=DEFAULT_VAL_RUNS,
                    help="How many runs to hold out for validation (threshold + early stopping)")
    ap.add_argument('--stride', type=int, default=1,
                    help="Stride between TRAINING windows (1 = original methodology; "
                         "values >1 reduce redundancy and speed things up)")
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--out', default=DEFAULT_OUTPUT_DIR,
                    help="Output folder. Use one subfolder per experiment "
                         "(e.g. models/ICE/task1) so runs do not overwrite each other.")
    args = ap.parse_args()

    set_output_dir(args.out)

    print("=" * 66)
    print(f"  MODEL RECALIBRATION (domain adaptation) -> {args.out}")
    print("=" * 66)

    print(f"\n1. Selecting normal runs from: {', '.join(args.data)}")
    files = collect_normal_files(args.data)
    if not files:
        print("[X] No normal run found.")
        sys.exit(1)
    per_file = load_raw_per_file(files)
    if not per_file:
        print("[X] No valid file (all discarded).")
        sys.exit(1)

    print(f"\n2. Per-run split (seed={args.seed})")
    train_files, val_files = split_runs(per_file, args.val_runs, args.seed)
    print(f"   TRAIN ({len(train_files)} run): {', '.join(os.path.basename(f) for f, _ in train_files)}")
    print(f"   VAL   ({len(val_files)} run): {', '.join(os.path.basename(f) for f, _ in val_files)}")

    print("\n3. Per-category scaler (fitted ONLY on training runs)")
    params = calculate_scaler(np.vstack([v for _, v in train_files]))
    write_scaler_json(params)

    print(f"\n4. Finestre per-file (stride training={args.stride}, validation=1)")
    X_train = np.concatenate([_windows(_scale(v, params), args.stride) for _, v in train_files])
    X_val = np.concatenate([_windows(_scale(v, params), 1) for _, v in val_files])
    print(f"   train: {X_train.shape}   val: {X_val.shape}   (device: {DEVICE})")

    selected = list(RECALIBRATORS) if 'all' in args.model else args.model
    print(f"\n5. Ricalibrazione: {', '.join(selected)}")
    results = {}
    for name in selected:
        results[name] = RECALIBRATORS[name](X_train, X_val, args)

    # The manifest is UPDATED, not rewritten: training one model at a time
    # (e.g. only the CNN with more epochs) must not erase the others' results.
    manifest = dict(models={})
    if os.path.exists(MANIFEST_OUTPUT):
        try:
            with open(MANIFEST_OUTPUT) as f:
                manifest = json.load(f)
            manifest.setdefault('models', {})
        except Exception:
            manifest = dict(models={})

    manifest.update(
        data_sources=args.data,
        # Task-qualified names (e.g. 'task1/run_02.csv'): runs with the same
        # basename exist under different tasks, the basename alone would be ambiguous
        # nella valutazione cross-task.
        train_runs=[run_label(f) for f, _ in train_files],
        val_runs=[run_label(f) for f, _ in val_files],
        n_windows_train=int(len(X_train)), n_windows_val=int(len(X_val)),
        scaler=params,
    )
    # Config and results are PER MODEL: each model may have been trained
    # con epoche/stride diversi, e va tracciato singolarmente.
    for name, res in results.items():
        manifest['models'][name] = dict(
            res, config=dict(epochs=args.epochs, patience=args.patience,
                             stride=args.stride, seed=args.seed,
                             batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                             window_size=WINDOW_SIZE))
    with open(MANIFEST_OUTPUT, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 66)
    print("  SUMMARY - held-out q99 threshold to use in the dashboard")
    print("=" * 66)
    for name in selected:
        r = results[name]
        flag = '' if r['converged'] else '   [!] UNDERTRAINED'
        print(f"   {name:12s} q99={r['q99']:.5f}  (val loss {r['val_loss']:.6f} @ epoch {r['best_epoch']}){flag}")
    print("\n   NB: val loss and thresholds are NOT comparable across models: CNN/LSTM/Transformer")
    print("       reconstruct the raw window (1200 values), the MLP the 95 statistics.")
    print("       Compare them on the ability to DETECT anomalies instead (next step).")
    print(f"\n   Reproducible manifest -> {MANIFEST_OUTPUT}")
    print(f"\n[DONE] Start the bridge with a model from {args.out}:")
    print(f"   python models/deploy/bridge.py --robot-ip <IP> --mode pc "
          f"--onnx-model {os.path.join(args.out, '<modello>_autoencoder.onnx')}")


if __name__ == '__main__':
    main()
