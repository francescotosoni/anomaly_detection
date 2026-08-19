"""
Evaluation of the recalibrated models on the lab anomaly recordings.

Evaluates one model (ONNX inside a models/ICE/<exp>/ folder) against the anomaly
files of a task, using the labels produced by label_lab_anomalies.py.

Two families of metrics, consistent with evaluate_metrics.py:

  [1] EVENT-BASED (industrial deployment) - with the MCU safety protocol:
      debouncing (5 consecutive windows above threshold) + persistence (250 windows).
      A physical anomaly counts as detected if the alarm fires during the event.
      Precision/Recall/F1 over EVENTS. Uses the 'primary' labels and the held-out
      q99 threshold saved during training. False alarms are counted on the NORMAL
      runs unseen in training (never on training data).

  [2] PER-WINDOW ROC-AUC / PR-AUC (model quality, threshold-independent):
      ability to separate normal from anomalous windows. Uses the 'tight' labels to
      avoid the artefact of persistent labels (paper_comparison.md).

Supports CROSS-TASK evaluation (model from one task, data from another) for the
generalisation study: the scaler used is ALWAYS the one belonging to the model.

Usage:
  python utils/evaluate_lab.py --model-dir models/ICE/task1 --eval-task 1
  python utils/evaluate_lab.py --model-dir models/ICE/task1 --eval-task 2   # cross-task
  python utils/evaluate_lab.py --model-dir models/ICE/task1 --eval-task 1 --models cnn
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
from utils.evaluate_metrics import (
    apply_debouncing, apply_alarm_persistence,
    DEBOUNCE_WINDOWS, PERSISTENCE_WINDOWS,
)
from sklearn.metrics import roc_auc_score, average_precision_score

WINDOW_SIZE = 50
N_FEATURES = 24
MODELS = {'cnn': 'cnn_autoencoder', 'cnn_up': 'cnn_up_autoencoder',
          'mlp': 'mlp_autoencoder',
          'lstm': 'lstm_autoencoder', 'transformer': 'transformer_autoencoder'}


# ---------------------------------------------------------
# Preprocessing (the MODEL's scaler, not refitted)
# ---------------------------------------------------------
def load_scaler(model_dir):
    with open(os.path.join(model_dir, 'scaler.json')) as f:
        return json.load(f)

def scale(vals, params):
    out = vals.copy()
    for cat, p in params.items():
        a, b = p['slice']
        out[:, a:b] = (out[:, a:b] - p['mean']) / p['std']
    return out

def make_windows(scaled):
    n = len(scaled) - WINDOW_SIZE + 1
    if n <= 0:
        return np.empty((0, WINDOW_SIZE, N_FEATURES), dtype=np.float32)
    return np.stack([scaled[i:i + WINDOW_SIZE] for i in range(n)]).astype(np.float32)

def onnx_scores(sess, X):
    """
    Per-window score. The project ONNX files are exported with a fixed batch of 1
    (the same artefact the bridge uses in deployment), so one window is inferred at
    a time: this measures EXACTLY the model that runs on the robot.
    """
    if len(X) == 0:
        return np.empty(0)
    name = sess.get_inputs()[0].name
    out = np.empty(len(X), dtype=np.float64)
    for i in range(len(X)):
        r = sess.run(None, {name: X[i:i + 1]})
        out[i] = float(np.asarray(r[0]).ravel()[0])
    return out

def windows_over_segment(a, b, n_win):
    """Range of window indices overlapping the sample segment [a,b]."""
    lo = max(0, a - WINDOW_SIZE + 1)
    hi = min(n_win - 1, b)
    return lo, hi

def count_alarm_segments(alarm):
    """Number of contiguous alarm segments (used to count false alarms)."""
    a = np.asarray(alarm, dtype=int)
    diff = np.diff(np.concatenate(([0], a, [0])))
    return int((diff == 1).sum())


# ---------------------------------------------------------
# Evaluation of a single model
# ---------------------------------------------------------
def protocol(pred, debounce_win, persistence_win):
    """
    Apply the DEPLOYMENT alarm protocol to the per-window predictions.

    Windows here have step=1 (one per sample, 8 ms at 125 Hz), so the parameters
    are expressed in windows and derived from the milliseconds configured in the
    dashboard. With persistence_win=0 no persistence is applied, which is the
    current dashboard behaviour (the alarm clears as soon as the score falls back).

    """
    out = apply_debouncing(pred, debounce_win) if debounce_win > 1 else list(pred)
    if persistence_win > 0:
        out = apply_alarm_persistence(out, persistence_win)
    return np.asarray(out)


# Columns of qd (joint velocities) inside the 24-feature vector.
QD_SLICE = slice(12, 18)

def window_speed(raw):
    """Mean window speed: norm of qd averaged over the 50 timesteps.

    The raw, UNSCALED values are used so the number stays in rad/s and readable.
    Aligned with make_windows(): one value per window, same order.
    """
    inst = np.linalg.norm(raw[:, QD_SLICE], axis=1)
    n_win = len(raw) - WINDOW_SIZE + 1
    if n_win <= 0:
        return np.empty(0)
    c = np.concatenate([[0.0], np.cumsum(inst)])
    return (c[WINDOW_SIZE:] - c[:-WINDOW_SIZE]) / WINDOW_SIZE

def gated_threshold(base_thr, speeds, speed_cut, slow_mult):
    """Per-window threshold: higher when the robot is nearly still.

    WHY: 95-99% of false alarms concentrate in the slowest quartile, while
    anomalies are spread uniformly over speed and, at low speed, still score around
    0.8 median. A single threshold must therefore choose between tolerating slow
    false alarms and losing sensitivity everywhere; two per-regime thresholds avoid
    the trade-off.
    """
    if speed_cut is None:
        return np.full(len(speeds), base_thr)
    return np.where(speeds < speed_cut, base_thr * slow_mult, base_thr)

def collect_scores(model_key, model_dir, eval_task, base, params, train_runs):
    """
    SINGLE pass over the data: compute and keep the scores of every file.

    Separated from threshold application because ONNX inference is the expensive
    part (tens of thousands of windows per file) while threshold and alarm protocol
    are vector operations. This way a sweep over N thresholds costs as much as a
    single evaluation instead of N times as much.
    """
    onnx_path = os.path.join(model_dir, f'{MODELS[model_key]}.onnx')
    thr_path = os.path.join(model_dir, f'{MODELS[model_key]}_q99_threshold.npy')
    sess = ort.InferenceSession(onnx_path)

    # --- NORMAL set for false alarms: runs unseen in training ---
    # Compared by qualified name (task/run): in cross-task no run of the evaluation
    # task appears among the training ones (names from different tasks do not
    # collide), so they are all used -- correct, the model has never seen them.
    normal_files = [f for f in resolve_csv_paths([os.path.join(base, f'task{eval_task}', 'run*.csv')])
                    if run_label(f) not in train_runs]
    normal_scores, normal_speeds = [], []
    for f in normal_files:
        raw = features_array(read_telemetry(f))
        s = onnx_scores(sess, make_windows(scale(raw, params)))
        if len(s):
            normal_scores.append(s)
            normal_speeds.append(window_speed(raw))

    # --- ANOMALY files: scores + event info + 'tight' labels for the ROC ---
    labels_primary, _ = load_labels(eval_task, base=base, variant='primary')
    labels_tight, _ = load_labels(eval_task, base=base, variant='tight')

    anom = []
    for name, info in labels_primary.items():
        path = os.path.join(base, f'task{eval_task}', name)
        raw = features_array(read_telemetry(path))
        s = onnx_scores(sess, make_windows(scale(raw, params)))
        n_win = len(s)
        if n_win == 0:
            continue
        mask_t = labels_tight[name]['mask']
        wl = np.array([1 if mask_t[i:i + WINDOW_SIZE].any() else 0 for i in range(n_win)])
        anom.append(dict(name=name, info=info, scores=s, tight=wl,
                         speeds=window_speed(raw)))

    return dict(
        saved_threshold=float(np.load(thr_path)[0]),
        normal_scores=normal_scores,
        normal_speeds=normal_speeds,
        n_normal_windows=sum(len(s) for s in normal_scores),
        anom=anom,
    )

def event_metrics(bundle, threshold, debounce_win, persistence_win,
                  speed_cut=None, slow_mult=1.0):
    """Apply threshold and alarm protocol to already computed scores.

    With speed_cut the threshold becomes per-window: base*slow_mult below that
    velocita', base sopra (vedi gated_threshold).
    """
    fp_events = fp_windows = 0
    for s, v in zip(bundle['normal_scores'], bundle['normal_speeds']):
        thr = gated_threshold(threshold, v, speed_cut, slow_mult)
        pred = protocol((s > thr).astype(int), debounce_win, persistence_win)
        fp_events += count_alarm_segments(pred)
        fp_windows += int(np.sum(pred))

    tp_events = fn_events = n_events_total = n_truncated = 0
    per_file = {}
    for e in bundle['anom']:
        s, info = e['scores'], e['info']
        thr = gated_threshold(threshold, e['speeds'], speed_cut, slow_mult)
        pred = protocol((s > thr).astype(int), debounce_win, persistence_win)

        # Per-event recall (excludes events truncated at start/end of recording)
        truncated = set(info.get('truncated_events', []))
        file_tp = file_fn = 0
        for i, (a, b) in enumerate(info['events']):
            if i in truncated:
                n_truncated += 1
                continue
            lo, hi = windows_over_segment(a, b, len(s))
            n_events_total += 1
            if pred[lo:hi + 1].any():
                tp_events += 1
                file_tp += 1
            else:
                fn_events += 1
                file_fn += 1
        per_file[e['name']] = dict(type=info['type'], events=len(info['events']),
                                   truncated=len(truncated), tp=file_tp, fn=file_fn)

    precision = tp_events / (tp_events + fp_events) if (tp_events + fp_events) else 0.0
    recall = tp_events / (tp_events + fn_events) if (tp_events + fn_events) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return dict(
        threshold=threshold,
        tp=tp_events, fn=fn_events, fp=fp_events,
        n_events=n_events_total, n_truncated=n_truncated,
        precision=precision, recall=recall, f1=f1,
        fp_events=fp_events, fp_windows=fp_windows,
        n_normal_windows=bundle['n_normal_windows'],
        per_file=per_file,
    )

def auc_metrics(bundle):
    """Per-window ROC-AUC / PR-AUC. Threshold-independent: computed once.

    Scores and labels are built in the SAME order (normals, then anomalies) to
    guarantee a one-to-one alignment.
    """
    anom_scores = [e['scores'] for e in bundle['anom']]
    anom_tight = [e['tight'] for e in bundle['anom']]

    score_parts = bundle['normal_scores'] + anom_scores
    label_parts = []
    if bundle['normal_scores']:
        label_parts.append(np.zeros(bundle['n_normal_windows'], dtype=int))
    label_parts.extend(anom_tight)
    all_scores = np.concatenate(score_parts) if score_parts else np.empty(0)
    all_labels = np.concatenate(label_parts) if label_parts else np.empty(0)
    assert len(all_scores) == len(all_labels), "score/label disallineati"
    try:
        two_classes = len(np.unique(all_labels)) > 1
        roc = roc_auc_score(all_labels, all_scores) if two_classes else float('nan')
        prauc = average_precision_score(all_labels, all_scores) if two_classes else float('nan')
    except Exception:
        roc = prauc = float('nan')

    return dict(roc_auc=float(roc), pr_auc=float(prauc),
                anomalous_window_frac=float(np.concatenate(anom_tight).mean()) if anom_tight else 0.0)

def evaluate_model(model_key, model_dir, eval_task, base, params, train_runs,
                   debounce_win=DEBOUNCE_WINDOWS, persistence_win=PERSISTENCE_WINDOWS):
    bundle = collect_scores(model_key, model_dir, eval_task, base, params, train_runs)
    out = event_metrics(bundle, bundle['saved_threshold'], debounce_win, persistence_win)
    out.update(auc_metrics(bundle))
    return out

def main():
    ap = argparse.ArgumentParser(description="Evaluation of ICE models on lab anomaly data")
    ap.add_argument('--model-dir', required=True, help="Folder with the ONNX files and thresholds (e.g. models/ICE/task1)")
    ap.add_argument('--eval-task', type=int, required=True, help="Task to evaluate on (data + labels)")
    ap.add_argument('--base', default='data/lab')
    ap.add_argument('--models', nargs='+', default=['cnn', 'mlp'], choices=list(MODELS))
    # Alarm protocol expressed in MILLISECONDS, to match what is configured in the
    # dashboard. Windows have step=1 => 1 window = 8 ms.
    ap.add_argument('--debounce-ms', type=float, default=100.0,
                    help="Consecutive milliseconds above threshold to raise the "
                         "alarm (must match the dashboard debounce)")
    ap.add_argument('--persistence-ms', type=float, default=0.0,
                    help="Minimum alarm duration once raised. 0 = same as the "
                         "current dashboard (the alarm clears when the score falls back)")
    ap.add_argument('--sweep', nargs='*', type=float, default=None,
                    metavar='MULT',
                    help="Repeat the evaluation at multiple thresholds, expressed as "
                         "multipliers of the saved q99 (default: 1 1.25 1.5 2 3 4). "
                         "Use it to pick the threshold from data rather than by feel. "
                         "Does not rewrite the results JSON.")
    ap.add_argument('--speed-gate', nargs='*', type=float, default=None,
                    metavar='CUT',
                    help="Compare the single threshold with a DOUBLE one per speed "
                         "regime: below CUT rad/s it is multiplied by --slow-mult. "
                         "Default cuts: 0.05 0.10 0.15 0.20. Rationale: 95-99%% of false "
                         "alarms concentrate in the slowest quartile, while anomalies "
                         "are spread uniformly over speed.")
    ap.add_argument('--slow-mult', nargs='*', type=float, default=None, metavar='M',
                    help="Threshold multipliers in the slow regime (default: 2 3 4 5)")
    ap.add_argument('--sweep-debounce', nargs='*', type=float, default=None,
                    metavar='MS',
                    help="With --sweep, cross the thresholds with several debounce "
                         "values (default: 40 100 250 500 1000) and print a "
                         "recall / false-alarm grid. Useful when false positives are not "
                         "isolated spikes but sustained excursions: there the choice is a "
                         "PAIR (threshold, debounce), not a threshold alone.")
    args = ap.parse_args()

    ms_per_window = 1000.0 / SAMPLING_HZ          # step=1 -> 8 ms per finestra
    debounce_win = max(1, int(round(args.debounce_ms / ms_per_window)))
    persistence_win = int(round(args.persistence_ms / ms_per_window))

    params = load_scaler(args.model_dir)

    manifest_path = os.path.join(args.model_dir, 'manifest.json')
    train_runs = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            train_runs = set(json.load(f).get('train_runs', []))

    model_task = os.path.basename(args.model_dir.rstrip('/\\'))
    cross = f"CROSS-TASK (model {model_task} on task{args.eval_task} data)" \
        if str(args.eval_task) not in model_task else "IN-DOMAIN"

    print("=" * 78)
    print(f"  EVALUATION - {cross}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Alarm protocol: debounce {args.debounce_ms:.0f} ms ({debounce_win} windows)"
          f"  |  persistence {args.persistence_ms:.0f} ms ({persistence_win} windows)")
    print("=" * 78)

    if args.speed_gate is not None:
        cuts = args.speed_gate if args.speed_gate else [0.05, 0.10, 0.15, 0.20]
        smults = args.slow_mult if args.slow_mult else [2.0, 3.0, 4.0, 5.0]
        for mk in args.models:
            bundle = collect_scores(mk, args.model_dir, args.eval_task, args.base,
                                    params, train_runs)
            thr = bundle['saved_threshold']
            print(f"\n  [{mk}]  soglia base = {thr:.5f}   debounce {args.debounce_ms:.0f} ms")

            r0 = event_metrics(bundle, thr, debounce_win, persistence_win)
            print(f"\n    REFERENCE - single threshold {thr:.5f}: "
                  f"recall {100*r0['recall']:.0f}%  false alarms {r0['fp']}")

            print(f"\n    double threshold - recall % / false alarms")
            hdr = ''.join(f"{'x%.0f' % m:>12s}" for m in smults)
            print(f"    {'cut':>8s}{hdr}")
            print("    " + "-" * (8 + 12 * len(smults)))
            for c in cuts:
                row = f"    {c:8.3f}"
                for m in smults:
                    r = event_metrics(bundle, thr, debounce_win, persistence_win,
                                      speed_cut=c, slow_mult=m)
                    row += f"{100*r['recall']:8.0f}%/{r['fp']:<3d}"
                print(row)
            print(f"\n    Events evaluated: {r0['n_events']}  |  "
                  f"normal windows: {r0['n_normal_windows']}")
            print(f"    Absolute slow threshold: "
                  + '  '.join(f"x{m:.0f}={thr*m:.5f}" for m in smults))
        print("\n  Exploratory analysis: the results JSON was NOT rewritten.")
        return

    if args.sweep is not None:
        mults = args.sweep if args.sweep else [1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
        for mk in args.models:
            bundle = collect_scores(mk, args.model_dir, args.eval_task, args.base,
                                    params, train_runs)
            base_thr = bundle['saved_threshold']
            aucs = auc_metrics(bundle)

            if args.sweep_debounce is not None:
                debs = args.sweep_debounce if args.sweep_debounce else [40, 100, 250, 500, 1000]
                # How high the normals reach and how low the anomalies go: if the
                # ranges overlap no threshold separates them, and the grid only
                # helps pick the least bad compromise.
                nrm = np.concatenate(bundle['normal_scores'])
                an = np.concatenate([e['scores'][e['tight'] == 1] for e in bundle['anom']
                                     if (e['tight'] == 1).any()])
                print(f"\n  [{mk}]  q99 = {base_thr:.5f}   ROC-AUC {aucs['roc_auc']:.4f}")
                print(f"    held-out normals : q99 {np.quantile(nrm, 0.99):.5f}  "
                      f"q99.9 {np.quantile(nrm, 0.999):.5f}  max {nrm.max():.5f}")
                print(f"    anomalous windows: q01 {np.quantile(an, 0.01):.5f}  "
                      f"median {np.median(an):.5f}  min {an.min():.5f}")
                print(f"\n    recall (%) / false alarms - rows: threshold, columns: debounce")
                hdr = ''.join(f"{d:>13.0f}ms" for d in debs)
                print(f"    {'thresh':>9s}{hdr}")
                print("    " + "-" * (9 + 15 * len(debs)))
                for m in mults:
                    row = f"    {base_thr * m:9.5f}"
                    for d in debs:
                        dw = max(1, int(round(d / ms_per_window)))
                        r = event_metrics(bundle, base_thr * m, dw, persistence_win)
                        row += f"{100 * r['recall']:9.0f}%/{r['fp']:<4d}"
                    print(row)
                print(f"\n    Events evaluated: {r['n_events']}  |  "
                      f"normal windows: {r['n_normal_windows']}")
                continue
            print(f"\n  [{mk}]  saved q99 = {base_thr:.5f}   "
                  f"ROC-AUC {aucs['roc_auc']:.4f} (threshold-independent)")
            print(f"  {'mult':>5s} {'thresh':>9s} {'TP':>3s} {'FN':>3s} {'FP':>4s} "
                  f"{'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'FP/1k windows':>15s}")
            print("  " + "-" * 68)
            for m in mults:
                r = event_metrics(bundle, base_thr * m, debounce_win, persistence_win)
                per_k = 1000.0 * r['fp_windows'] / r['n_normal_windows'] if r['n_normal_windows'] else 0.0
                print(f"  {m:5.2f} {r['threshold']:9.5f} {r['tp']:3d} {r['fn']:3d} {r['fp']:4d} "
                      f"{r['precision']:6.3f} {r['recall']:7.3f} {r['f1']:6.3f} {per_k:15.2f}")
            print(f"\n  Events evaluated: {r['n_events']}  |  "
                  f"normal windows: {r['n_normal_windows']}")
        print("\n  Exploratory sweep: the results JSON was NOT rewritten.")
        return

    results = {}
    for mk in args.models:
        results[mk] = evaluate_model(mk, args.model_dir, args.eval_task, args.base, params,
                                     train_runs, debounce_win, persistence_win)

    # --- Summary table ---
    print(f"\n  {'model':10s} {'thresh':>8s} {'TP':>3s} {'FN':>3s} {'FP':>4s} "
          f"{'Prec':>6s} {'Recall':>7s} {'F1':>6s} {'ROC-AUC':>8s} {'PR-AUC':>7s}")
    print("  " + "-" * 74)
    for mk in args.models:
        r = results[mk]
        print(f"  {mk:10s} {r['threshold']:8.4f} {r['tp']:3d} {r['fn']:3d} {r['fp']:4d} "
              f"{r['precision']:6.3f} {r['recall']:7.3f} {r['f1']:6.3f} {r['roc_auc']:8.4f} {r['pr_auc']:7.4f}")

    print(f"\n  Events evaluated: {results[args.models[0]]['n_events']} "
          f"({results[args.models[0]]['n_truncated']} truncated excluded)  |  "
          f"normal windows (false alarms): {results[args.models[0]]['n_normal_windows']}")

    print("\n  Per-file detail (event-based recall):")
    for mk in args.models:
        print(f"    [{mk}]")
        for name, pf in results[mk]['per_file'].items():
            det = f"{pf['tp']}/{pf['events'] - pf['truncated']}"
            print(f"      {name:14s} [{pf['type']:11s}] detected {det} events"
                  + (f"  (+{pf['truncated']} truncated excluded)" if pf['truncated'] else ""))

    out = os.path.join(args.model_dir, f'eval_on_task{args.eval_task}.json')
    with open(out, 'w') as f:
        json.dump({mk: {k: v for k, v in r.items() if k != 'per_file'} for mk, r in results.items()},
                  f, indent=2)
    print(f"\n  [OK] Results -> {out}")

if __name__ == '__main__':
    main()
