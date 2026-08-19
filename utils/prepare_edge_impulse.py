"""
Prepare the Edge Impulse assets from the model recalibrated in the lab.

WHY IT IS NEEDED: the assets already in models/deploy/ (ei_test_data/, the .zip
v2/v3/v4 libraries) were generated from the PRE-recalibration models, scaled with
the academic dataset scaler. Reusing them on the ESP32 would bring back exactly the
domain shift that recalibration removed.

What it produces (in models/deploy/ei_<exp>/):
  * test_data/normal.NNN.csv and anomaly.NNN.csv - windows in the Edge Impulse CSV
    format (timestamp in ms + 24 columns), SCALED with the model's scaler. They are
    used on EI to validate that the imported model gives the same scores.

  * representative.npy - normal windows for int8 quantisation calibration.

  * expected_scores.csv - reference scores computed with onnxruntime on each
    window: the ground truth to compare the ESP32 output against.
  * README.md - import instructions.

The normal windows come ONLY from validation runs (held out): using training
windows would invalidate the check.

Usage:
  python utils/prepare_edge_impulse.py --model-dir models/ICE/combined --model mlp
  python utils/prepare_edge_impulse.py --model-dir models/ICE/combined --model mlp --n-normal 60 --n-anomaly 60
"""

import os
import sys
import json
import argparse

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.lab_io import read_telemetry, features_array, resolve_csv_paths, run_label, SAMPLING_HZ
from utils.label_lab_anomalies import load_labels

WINDOW_SIZE = 50
N_FEATURES = 24
MODEL_FILES = {'cnn': 'cnn_autoencoder', 'cnn_up': 'cnn_up_autoencoder',
               'mlp': 'mlp_autoencoder',
               'lstm': 'lstm_autoencoder', 'transformer': 'transformer_autoencoder'}

def scale(vals, params):
    out = vals.copy()
    for cat, p in params.items():
        a, b = p['slice']
        out[:, a:b] = (out[:, a:b] - p['mean']) / p['std']
    return out

def windows(scaled):
    n = len(scaled) - WINDOW_SIZE + 1
    if n <= 0:
        return np.empty((0, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
    return np.stack([scaled[i:i + WINDOW_SIZE] for i in range(n)]).astype(np.float32)

def onnx_score_one(sess, name, w):
    return float(np.asarray(sess.run(None, {name: w[None, :, :]})[0]).ravel()[0])

def write_ei_csv(path, window):
    """
    Edge Impulse CSV format for multi-axis time-series data:
      first column 'timestamp' in milliseconds, then one axis per column.
      Passo temporale = 1000/125 = 8 ms.
    """
    step_ms = int(round(1000 / SAMPLING_HZ))
    header = 'timestamp,' + ','.join(f'f{i}' for i in range(N_FEATURES))
    lines = [header]
    for t in range(window.shape[0]):
        vals = ','.join(f'{v:.6f}' for v in window[t])
        lines.append(f'{t * step_ms},{vals}')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')

def main():
    ap = argparse.ArgumentParser(description="Prepare Edge Impulse assets from a lab model")
    ap.add_argument('--model-dir', required=True, help="e.g. models/ICE/combined")
    ap.add_argument('--model', default='mlp', choices=list(MODEL_FILES))
    ap.add_argument('--base', default='data/lab')
    ap.add_argument('--tasks', nargs='+', type=int, default=[1, 2])
    ap.add_argument('--n-normal', type=int, default=50, help="Normal windows to export")
    ap.add_argument('--n-anomaly', type=int, default=50, help="Anomalous windows to export")
    ap.add_argument('--n-representative', type=int, default=500,
                    help="Windows for the int8 calibration set")
    ap.add_argument('--score-cap', type=float, default=None, metavar='MULT',
                    help="Extend the calibration set with ANOMALOUS windows whose "
                         "float32 score reaches up to MULT times the threshold. "
                         "WHY: Edge Impulse derives the int8 output range from the "
                         "calibration data; with normals only the range stops just "
                         "above the threshold (measured 1.05x) and every anomaly "
                         "saturates, which prevents both reading the severity and "
                         "RAISING the threshold above q99 -- the quantile would land "
                         "on full scale. Reasonable value: 4. Note: the higher it is, "
                         "the coarser the resolution near the threshold.")
    ap.add_argument('--rep-anomaly-frac', type=float, default=0.2, metavar='F',
                    help="With --score-cap, the fraction of the calibration set taken "
                         "from anomalous windows (default 0.2). The rest stays normal: "
                         "the normal band must remain well represented, that is where "
                         "the decision falls.")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    with open(os.path.join(args.model_dir, 'scaler.json')) as f:
        params = json.load(f)
    with open(os.path.join(args.model_dir, 'manifest.json')) as f:
        manifest = json.load(f)
    val_runs = set(manifest.get('val_runs', []))

    stem = MODEL_FILES[args.model]
    onnx_path = os.path.join(args.model_dir, f'{stem}.onnx')
    threshold = float(np.load(os.path.join(args.model_dir, f'{stem}_q99_threshold.npy'))[0])
    sess = ort.InferenceSession(onnx_path)
    in_name = sess.get_inputs()[0].name

    exp = os.path.basename(args.model_dir.rstrip('/\\'))
    out_dir = os.path.join('models', 'deploy', f'ei_{exp}_{args.model}')
    data_dir = os.path.join(out_dir, 'test_data')
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 70)
    print(f"  EDGE IMPULSE ASSETS - {args.model.upper()} model from {args.model_dir}")
    print("=" * 70)
    print(f"  ONNX: {onnx_path}")
    print(f"  q99 threshold: {threshold:.5f}")
    print(f"  Scaler: the model's own (NOT refitted)")

    rng = np.random.RandomState(args.seed)

    # ---------- NORMAL windows: validation runs only (held out) ----------
    normal_windows = []
    for t in args.tasks:
        for f in resolve_csv_paths([os.path.join(args.base, f'task{t}', 'run*.csv')]):
            if run_label(f) not in val_runs:
                continue
            normal_windows.append(windows(scale(features_array(read_telemetry(f)), params)))
    if not normal_windows:
        print("[X] No validation run found: check the manifest.")
        sys.exit(1)
    normal_all = np.concatenate(normal_windows)
    print(f"\n  Normal windows available (held-out val): {len(normal_all)}")

    # ---------- ANOMALOUS windows: where the 'tight' label is active ----------
    anom_windows = []
    for t in args.tasks:
        try:
            labels, _ = load_labels(t, base=args.base, variant='tight')
        except FileNotFoundError:
            print(f"  [!] label del task {t} assenti -> salto")
            continue
        for name, info in labels.items():
            path = os.path.join(args.base, f'task{t}', name)
            X = windows(scale(features_array(read_telemetry(path)), params))
            if len(X) == 0:
                continue
            mask = info['mask']
            keep = [i for i in range(len(X)) if mask[i:i + WINDOW_SIZE].all()]
            if keep:
                anom_windows.append(X[keep])
    if not anom_windows:
        print("[X] No anomalous window found.")
        sys.exit(1)
    anom_all = np.concatenate(anom_windows)
    print(f"  Anomalous windows available (full tight label): {len(anom_all)}")

    # ---------- Sampling and export ----------
    n_norm = min(args.n_normal, len(normal_all))
    n_anom = min(args.n_anomaly, len(anom_all))
    idx_n = rng.choice(len(normal_all), n_norm, replace=False)
    idx_a = rng.choice(len(anom_all), n_anom, replace=False)

    rows = []
    for k, i in enumerate(idx_n):
        w = normal_all[i]
        fn = f'normal.{k:03d}.csv'
        write_ei_csv(os.path.join(data_dir, fn), w)
        rows.append((fn, 'normal', onnx_score_one(sess, in_name, w)))
    for k, i in enumerate(idx_a):
        w = anom_all[i]
        fn = f'anomaly.{k:03d}.csv'
        write_ei_csv(os.path.join(data_dir, fn), w)
        rows.append((fn, 'anomaly', onnx_score_one(sess, in_name, w)))

    with open(os.path.join(out_dir, 'expected_scores.csv'), 'w') as f:
        f.write('file,label,expected_score,exceeds_threshold\n')
        for fn, lab, sc in rows:
            f.write(f'{fn},{lab},{sc:.8f},{int(sc > threshold)}\n')

    ns = np.array([r[2] for r in rows if r[1] == 'normal'])
    as_ = np.array([r[2] for r in rows if r[1] == 'anomaly'])
    print(f"\n  Exported {n_norm} normal + {n_anom} anomalous windows to {data_dir}")
    print(f"    normal scores: median={np.median(ns):.5f}  max={ns.max():.5f}  "
          f"above threshold={int((ns > threshold).sum())}/{n_norm}")
    print(f"    anomaly scores: median={np.median(as_):.5f}  min={as_.min():.5f}  "
          f"above threshold={int((as_ > threshold).sum())}/{n_anom}")

    # ---------- Representative data for int8 quantisation ----------
    # Saved in TWO forms: Edge Impulse sometimes wants the (n,50,24) tensor and
    # sometimes the flattened features (n,1200). Flattening is timestep-major, the
    # same order in which the firmware fills inference_buffer.
    n_rep = min(args.n_representative, len(normal_all))

    if args.score_cap is None:
        rep = normal_all[rng.choice(len(normal_all), n_rep, replace=False)]
        rep_note = "normal windows only"
    else:
        # MILD anomalies are needed, not the most violent ones: the output range
        # must reach ~cap times the threshold, not 6.4 (the observed maximum), which
        # would make 1 LSB larger than the whole normal band.
        cap = args.score_cap * threshold
        n_a = int(round(n_rep * args.rep_anomaly_frac))
        n_n = n_rep - n_a

        # Sample a subset and score it: evaluating them all would cost tens of
        # thousands of inferences for a handful of useful windows.
        pool = anom_all[rng.choice(len(anom_all), min(4000, len(anom_all)), replace=False)]
        sc = np.array([onnx_score_one(sess, in_name, w) for w in pool])
        eligible = np.where((sc > threshold) & (sc <= cap))[0]
        print(f"\n  Anomalous windows with score in ({threshold:.5f}, {cap:.5f}]: "
              f"{len(eligible)}/{len(pool)} sampled")
        if len(eligible) == 0:
            print(f"  [X] No anomaly below {cap:.5f}. Raise --score-cap: the model "
                  f"separates so well that no mild anomaly exists in this range.")
            sys.exit(1)
        if len(eligible) < n_a:
            print(f"  [!] Only {len(eligible)} available instead of {n_a}: using those.")
            n_a = len(eligible)
            n_n = n_rep - n_a
        take_a = pool[rng.choice(eligible, n_a, replace=False)]
        take_n = normal_all[rng.choice(len(normal_all), n_n, replace=False)]
        rep = np.concatenate([take_n, take_a])
        rng.shuffle(rep)
        rep_note = (f"{n_n} normal + {n_a} anomalous with score <= {cap:.5f} "
                    f"({args.score_cap:g}x the threshold)")
        print(f"  Expected output range after quantisation: ~0 .. {cap:.5f} "
              f"(1 LSB ~ {cap / 255:.6f}, threshold at ~{threshold / (cap / 255):.0f} LSB)")

    np.save(os.path.join(out_dir, 'representative.npy'), rep)
    np.save(os.path.join(out_dir, 'representative_flat.npy'), rep.reshape(len(rep), -1))
    print(f"\nRepresentative data for int8: {rep_note}")
    print(f"    representative.npy      {rep.shape}   (tensor, same as the ONNX input)")
    print(f"    representative_flat.npy {(len(rep), rep.shape[1] * rep.shape[2])}   "
          f"(flattened timestep-major, the usual EI format)")

    # ---------- README ----------
    with open(os.path.join(out_dir, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# Edge Impulse assets - {args.model.upper()} ({exp})

Generated by `utils/prepare_edge_impulse.py` from the model recalibrated in the lab.

| | |
|---|---|
| ONNX to import | `{onnx_path}` |
| q99 threshold (held-out) | **{threshold:.5f}** |
| Model input | `(1, {WINDOW_SIZE}, {N_FEATURES})` float32 |
| Output | scalar = anomaly score (reconstruction MSE) |
| Sampling rate | {SAMPLING_HZ} Hz (step {int(round(1000 / SAMPLING_HZ))} ms) |

## Important

**Scaling stays on the PC**: `bridge.py --mode esp32` sends already scaled features.
The ESP32 does not need to know the scaler, it only runs the model.

The normal windows in `test_data/` come **only from validation runs** (held out
from training), so the check is honest.

## Procedure

1. Edge Impulse Studio -> new project -> **Bring Your Own Model (BYOM)**.
   Load `{stem}.onnx`. The graph already contains feature engineering + MSE, so it
   must be imported **whole**: the output is a single scalar, hence a 1-output
   **regression** and not a native EI Anomaly Detection block.
2. Target device: the board actually in use (ESP32-S3 for the current deployment).
3. int8 quantisation: use `representative.npy` as the calibration dataset.
   Compare the int8 scores with `expected_scores.csv` - if they diverge a lot,
   keep it in float32.
4. Deployment -> **Arduino library** -> download the .zip.
5. Extract into `Documents/Arduino/libraries/`, update the `#include` line in
   `models/deploy/firmware/firmware.ino`.
6. Arduino IDE -> select your board -> Upload.
7. Acceptance check:
   `python utils/verify_mcu.py --serial-port COMx --assets {out_dir}`
8. Set `anomaly_threshold = {threshold:.5f}f` in the firmware.

## Files

- `test_data/normal.NNN.csv`, `test_data/anomaly.NNN.csv` - windows in EI CSV format
- `expected_scores.csv` - reference scores (onnxruntime): the ground truth to compare against
- `representative.npy` - {n_rep} normal windows for int8 calibration, shape ({n_rep}, {WINDOW_SIZE}, {N_FEATURES})
- `representative_flat.npy` - the same, flattened to ({n_rep}, {WINDOW_SIZE * N_FEATURES}): try this if EI rejects the 3D tensor
""")
    print(f"  Instructions -> {out_dir}/README.md")
    print(f"\n[OK] Assets ready in {out_dir}")

if __name__ == '__main__':
    main()
