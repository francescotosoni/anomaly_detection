"""
Lazy access to the nominal window array for the sequential models.

The window array grew to 1.53 M windows (7.3 GB in float32) once the branch runs
entered the training set. The previous pattern - np.load, train_test_split, then
torch.tensor - holds three copies at once and does not fit in 14.7 GB of RAM.

Here the array stays memory-mapped on disk and windows are materialised one
batch at a time, with the feature mask applied on the fly. Only the split
indices live in memory (a few MB).

The split is by index with the same proportions and seed as before, so that all
models see the same partition: 70% train, 10% validation, 20% held-out test.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.feature_engineering import apply_feature_mask


class WindowDataset(Dataset):
    """Autoencoder dataset over a memory-mapped window array.

    Returns (x, x): the target is the input, as the models reconstruct it.
    `channels_first` transposes to (features, time) for Conv1d.
    """

    def __init__(self, path, indices, model_type="sequential", channels_first=False):
        self.path = path
        self.indices = np.asarray(indices)
        self.model_type = model_type
        self.channels_first = channels_first
        self._array = None  # opened lazily, so worker processes get their own handle

    @property
    def array(self):
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        w = np.asarray(self.array[self.indices[i]])
        w = apply_feature_mask(w[None, ...], model_type=self.model_type)[0]
        x = torch.from_numpy(np.ascontiguousarray(w)).float()
        if self.channels_first:
            x = x.transpose(0, 1)
        return x, x

    @property
    def n_features(self):
        probe = apply_feature_mask(np.zeros((1, 1, self.array.shape[-1])),
                                   model_type=self.model_type)
        return probe.shape[-1]


def split_indices(n, seed=42, test_frac=0.20, val_frac_of_rest=0.125):
    """70 / 10 / 20 split over window indices, matching the historical seed."""
    perm = np.random.RandomState(seed).permutation(n)
    n_test = int(np.ceil(n * test_frac))
    idx_test = np.sort(perm[:n_test])
    rest = perm[n_test:]
    n_val = int(np.ceil(len(rest) * val_frac_of_rest))
    return np.sort(rest[n_val:]), np.sort(rest[:n_val]), idx_test


def write_subset(path_in, indices, path_out, chunk=20000):
    """Copy the selected windows to a new .npy without loading the source."""
    src = np.load(path_in, mmap_mode="r")
    out = np.lib.format.open_memmap(path_out, mode="w+", dtype=src.dtype,
                                    shape=(len(indices),) + src.shape[1:])
    for s in range(0, len(indices), chunk):
        out[s:s + chunk] = src[indices[s:s + chunk]]
    out.flush()
    del out


def prepare(data_path, suffix, test_path=None, seed=42, model_type="sequential",
            channels_first=False):
    """Open the array, split it, and return (train_ds, val_ds, n_windows_test).

    When `test_path` is given and missing on disk, the held-out split is written
    there so the evaluation script can reuse it.
    """
    arr = np.load(data_path, mmap_mode="r")
    idx_train, idx_val, idx_test = split_indices(len(arr), seed=seed)

    if test_path and not os.path.exists(test_path):
        print(f"   writing held-out nominal test split to {test_path}")
        write_subset(data_path, idx_test, test_path)

    train_ds = WindowDataset(data_path, idx_train, model_type, channels_first)
    val_ds = WindowDataset(data_path, idx_val, model_type, channels_first)
    return train_ds, val_ds, len(idx_test)


def build_args(model_name):
    """Common CLI for the training scripts."""
    ap = argparse.ArgumentParser(description=f'Train the {model_name} autoencoder on nominal windows')
    ap.add_argument('--data', default=None,
                    help='window array to train on (default: data/X_normal<suffix>.npy)')
    ap.add_argument('--suffix', default='',
                    help='appended to every input and output name, e.g. _v2')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0,
                    help='DataLoader workers; 0 keeps the memmap in this process')
    return ap.parse_args()
