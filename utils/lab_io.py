"""
Shared I/O for telemetry recorded in the lab.

Single place where the 24 features are defined and a telemetry CSV is read.
The separator is DETECTED automatically because two formats coexist:
  - space  : legacy files (RTDE library default, as in the Graabaek dataset)
  - comma  : new files produced by collect.py (StateRecorder, delimiter=',')
"""

import os
import glob

import numpy as np
import pandas as pd

# Canonical order of the 24 physical features (must match anomaly_config_24.xml)
FEATURES = [
    'actual_q_0', 'actual_q_1', 'actual_q_2', 'actual_q_3', 'actual_q_4', 'actual_q_5',
    'actual_current_0', 'actual_current_1', 'actual_current_2', 'actual_current_3', 'actual_current_4', 'actual_current_5',
    'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 'actual_qd_3', 'actual_qd_4', 'actual_qd_5',
    'actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2', 'actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5',
]

# Useful subsets
FORCE_COLS = ['actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2']
TORQUE_COLS = ['actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5']
QD_COLS = [f'actual_qd_{i}' for i in range(6)]

SAMPLING_HZ = 125


def detect_separator(path):
    """Detect the separator from the header: comma if present, space otherwise."""
    with open(path, 'r') as f:
        header = f.readline()
    return ',' if header.count(',') >= 2 else ' '


def read_telemetry(path):
    """
    Read a telemetry CSV with auto-detected separator.
    Returns the full DataFrame (including 'timestamp' if present).
    """
    return pd.read_csv(path, sep=detect_separator(path))


def features_array(df):
    """Extract the 24 features in canonical order -> (M, 24) float64 array."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df[FEATURES].dropna().values.astype(np.float64)


def force_magnitude(df):
    """TCP force magnitude (N) for every sample -> (M,) array."""
    return np.linalg.norm(df[FORCE_COLS].values, axis=1)


def run_label(path):
    """
    Task-qualified label: 'task1/run_10.csv'. Needed because runs with the same
    base name exist under different tasks (e.g. run_10.csv in both task1 and
    task2): the basename alone would be ambiguous in manifests and evaluation.
    """
    p = os.path.abspath(path)
    return os.path.basename(os.path.dirname(p)) + '/' + os.path.basename(p)


def resolve_csv_paths(paths):
    """Expand files, directories (-> *.csv) and globs into an ordered, deduplicated list."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.csv')))
        elif any(c in p for c in '*?['):
            files += sorted(glob.glob(p))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"   [!] path not found, skipped: {p}")
    seen, out = set(), []
    for f in files:
        fa = os.path.abspath(f)
        if fa not in seen:
            seen.add(fa)
            out.append(f)
    return out
