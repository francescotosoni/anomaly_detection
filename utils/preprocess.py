"""
Turn the raw stochastic_PnP CSV runs into the sliding-window arrays the models
train on, plus the fitted scaler.

NOMINAL DIRECTORIES
-------------------
The dataset splits the nominal runs across three folders. Measuring the longest
stationary stretch of each run tells them apart:

    normal             508 runs   longest pause 0.86 - 0.96 s   never enters the branch
    normal_branch      102 runs   longest pause 5.4 - 10.9 s    the [4,10] s wait at node 2
    normal_extra_wait   35 runs   longest pause 20.7 - 21.0 s   longer than the long_wait anomaly

508 + 102 = 610, and 102/610 = 0.167, exactly the branch probability stated in
the paper - so `normal` and `normal_branch` together are the ~600 nominal runs
Graabaek et al. train on. Earlier versions of this script used `normal` alone,
which left the rarely-visited program branch entirely out of training while
roughly 45% of the anomalous evaluation runs traverse it.

`normal_extra_wait` is deliberately excluded: a 21 s pause is longer than the
long_wait anomaly itself (12 / 15 / 20 s), so training on it would teach the
model to accept the very behaviour long_wait is meant to flag.

MEMORY
------
With 610 runs the window array is about 13 GB in float64, which does not fit
alongside the intermediate list. Windows are written straight into a memmap in
float32 - the dtype the models cast to anyway, so nothing is lost.

Usage:
    python utils/preprocess.py --suffix _v2
    python utils/preprocess.py --dirs normal            # previous behaviour
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd


class CustomCategoryScaler:
    def __init__(self):
        # 6 joint positions (0-5), 6 currents (6-11), 6 velocities (12-17), 6 forces (18-23)
        self.categories = {
            'q': slice(0, 6),
            'current': slice(6, 12),
            'qd': slice(12, 18),
            'tcp': slice(18, 24)
        }
        self.means = {}
        self.stds = {}

    def fit(self, X):
        for cat, slc in self.categories.items():
            cat_data = X[:, slc]
            self.means[cat] = np.mean(cat_data)
            self.stds[cat] = np.std(cat_data)
            if self.stds[cat] == 0:
                self.stds[cat] = 1.0
        return self

    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for cat, slc in self.categories.items():
            X_scaled[:, slc] = (X[:, slc] - self.means[cat]) / self.stds[cat]
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "data", "stochastic_PnP")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Nominal folders used for training. See the module docstring for why
# normal_extra_wait is not among them.
NOMINAL_DIRS = ["normal", "normal_branch"]

ANOMALY_TYPES = [
    "change_weight",
    "drop_object",
    "force_stop",
    "long_wait",
    "outside_box",
    "push",
    "scale_speed",
    "tcp_wrench",
]

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

WINDOW_SIZE = 50
STEP_SIZE = 1
DTYPE = np.float32


def load_and_filter_data(filepath, sep=' '):
    df = pd.read_csv(filepath, sep=sep)
    df = df[FEATURES].dropna()
    return df.values


def collect_runs(directories):
    """Read every run of the listed folders and return the raw arrays.

    Raw traces are small (~150 MB for all 610 runs); it is the windowing that
    explodes, so it is safe to hold these in memory.
    """
    runs = []
    for d in directories:
        path = os.path.join(BASE_DIR, d)
        files = sorted(glob.glob(os.path.join(path, "*.csv")) +
                       glob.glob(os.path.join(path, "test", "*.csv")))
        kept = 0
        for f in files:
            data = load_and_filter_data(f)
            if len(data) >= WINDOW_SIZE:
                runs.append(data)
                kept += 1
        print(f"  -> {d}: {kept}/{len(files)} runs kept")
    return runs


def n_windows(run_len):
    return max(0, (run_len - WINDOW_SIZE) // STEP_SIZE + 1)


def write_windows(runs, scaler, out_path):
    """Scale each run and stream its windows into a memmap on disk."""
    total = sum(n_windows(len(r)) for r in runs)
    shape = (total, WINDOW_SIZE, len(FEATURES))
    print(f"  -> writing {total:,} windows {shape} as {np.dtype(DTYPE).name} "
          f"({np.prod(shape) * np.dtype(DTYPE).itemsize / 1e9:.1f} GB)")

    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=DTYPE, shape=shape)
    pos = 0
    for r in runs:
        k = n_windows(len(r))
        if k == 0:
            continue
        scaled = scaler.transform(r).astype(DTYPE, copy=False)
        # sliding_window_view gives a view; the copy happens on assignment
        view = np.lib.stride_tricks.sliding_window_view(scaled, WINDOW_SIZE, axis=0)
        out[pos:pos + k] = view[::STEP_SIZE].transpose(0, 2, 1)
        pos += k
    out.flush()
    del out
    return shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", default=NOMINAL_DIRS,
                    help="nominal folders used for training and for fitting the scaler")
    ap.add_argument("--suffix", default="",
                    help="appended to the output names, e.g. _v2 -> data/X_normal_v2.npy")
    ap.add_argument("--skip-anomalies", action="store_true",
                    help="only regenerate the nominal array and the scaler")
    args = ap.parse_args()

    sfx = args.suffix
    x_normal = os.path.join(OUT_DIR, f"X_normal{sfx}.npy")
    scaler_path = os.path.join(OUT_DIR, f"scaler{sfx}.pkl")

    print(f"1. Nominal baseline from {', '.join(args.dirs)}")
    runs = collect_runs(args.dirs)
    if not runs:
        raise SystemExit("no nominal run found")

    print("  -> fitting the scaler on all nominal data")
    scaler = CustomCategoryScaler()
    scaler.fit(np.vstack(runs))
    joblib.dump(scaler, scaler_path)
    print(f"     saved {scaler_path}")
    for cat in scaler.categories:
        print(f"     {cat:8s} mean {scaler.means[cat]:+.4f}  std {scaler.stds[cat]:.4f}")

    shape = write_windows(runs, scaler, x_normal)
    print(f"     saved {x_normal} {shape}")
    del runs

    if args.skip_anomalies:
        return

    print("\n2. Anomaly categories")
    for anomaly in ANOMALY_TYPES:
        anom_runs = collect_runs([anomaly])
        if not anom_runs:
            print(f"   [empty] {anomaly}")
            continue
        out = os.path.join(OUT_DIR, f"X_anomaly_{anomaly}{sfx}.npy")
        shape = write_windows(anom_runs, scaler, out)
        print(f"   [saved] {anomaly}: {shape}")

    print("\n[OK] datasets extracted and normalised")


if __name__ == "__main__":
    main()
