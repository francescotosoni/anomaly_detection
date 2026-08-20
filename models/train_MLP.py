import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_engineering import apply_feature_mask, extract_tabular_features

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
BOTTLENECK_DIM = 32

# --- COMPOSITE LOSS (MSE + Cosine Similarity) ---
class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.alpha = alpha
    
    def forward(self, x, target):
        mse_loss = self.mse(x, target)
        cos_loss = (1 - self.cosine(x, target)).mean()
        return self.alpha * mse_loss + (1 - self.alpha) * cos_loss

# --- MODEL DEFINITION (MLP Autoencoder Feature Engineered) ---
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=32):
        super(MLPAutoencoder, self).__init__()
        self.input_dim = input_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, bottleneck_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def tabular_chunked(X, idx, chunk=20000):
    """Feature-mask + statistics for the given window indices, in chunks.

    The window array is several GB; the 95-dimensional tabular representation is
    a few hundred MB. Materialising the windows for a whole split would need
    twice the array in RAM, so the reduction is applied chunk by chunk and only
    the result is kept.
    """
    out = []
    for s in range(0, len(idx), chunk):
        block = np.asarray(X[idx[s:s + chunk]])
        out.append(extract_tabular_features(apply_feature_mask(block, model_type='tabular')))
    return np.concatenate(out) if out else np.empty((0, 0))


def write_subset(X, idx, path, chunk=20000):
    """Copy the selected windows into a .npy on disk without loading the source."""
    out = np.lib.format.open_memmap(path, mode='w+', dtype=X.dtype,
                                    shape=(len(idx),) + X.shape[1:])
    for s in range(0, len(idx), chunk):
        out[s:s + chunk] = X[idx[s:s + chunk]]
    out.flush()
    del out


def main():
    parser = argparse.ArgumentParser(description='Train the MLP autoencoder on nominal windows')
    parser.add_argument('--data', default=None,
                        help='window array to train on (default: data/X_normal<suffix>.npy)')
    parser.add_argument('--suffix', default='',
                        help='appended to every input and output name, e.g. _v2')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    sfx = args.suffix
    data_path = args.data or f'data/X_normal{sfx}.npy'
    model_path = f'models/mlp_autoencoder{sfx}.pth'
    thr_path = f'models/mlp{sfx}_q99_threshold.npy'
    test_path = f'data/X_test_normal{sfx}.npy'

    print('1. Loading Data & Splitting (70% Train, 10% Val, 20% Test)...')
    print(f'   source: {data_path}')
    # Memory-mapped: the array is read through the split indices, never as a whole.
    X_normal_all = np.load(data_path, mmap_mode='r')
    n = len(X_normal_all)

    # Same proportions and seed as before, expressed on indices so that the
    # windows themselves are never duplicated in RAM.
    perm = np.random.RandomState(args.seed).permutation(n)
    n_test = int(np.ceil(n * 0.20))
    idx_test = np.sort(perm[:n_test])
    rest = perm[n_test:]
    n_val = int(np.ceil(len(rest) * 0.125))
    idx_val = np.sort(rest[:n_val])
    idx_train = np.sort(rest[n_val:])

    print(f'   writing held-out nominal test split to {test_path}')
    write_subset(X_normal_all, idx_test, test_path)

    X_train = tabular_chunked(X_normal_all, idx_train)
    X_val = tabular_chunked(X_normal_all, idx_val)

    input_dim = X_train.shape[1]

    print(f'   Healthy Train: {X_train.shape[0]} | Healthy Val: {X_val.shape[0]} | Healthy Test: {len(idx_test)}')
    print(f'   Input Dim: {input_dim} (19 features * 5 statistics)')

    X_train_pt = torch.tensor(X_train, dtype=torch.float32)
    X_val_pt = torch.tensor(X_val, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_pt, X_train_pt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_pt, X_val_pt), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'   Using device: {device}')
    
    model = MLPAutoencoder(input_dim=input_dim, bottleneck_dim=BOTTLENECK_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print('\n2. Training Model (MLP Feature Engineered)...')
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 15
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()
        
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_data.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, _ in val_loader:
                batch_data = batch_data.to(device)
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_data)
                val_loss += loss.item() * batch_data.size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        elapsed = time.time() - start_time
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'   Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.2f}s')
            
        if patience_counter >= PATIENCE:
            print(f'   Early stopping at epoch {epoch+1}')
            break
            
    print('\n3. Calculating Threshold (Q99) on Validation Set...')
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    val_errors = []
    mse_criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        for batch_data, _ in val_loader:
            batch_data = batch_data.to(device)
            reconstructed = model(batch_data)
            # MSE per sample
            loss_per_sample = mse_criterion(reconstructed, batch_data).mean(dim=1).cpu().numpy()
            val_errors.extend(loss_per_sample)
            
    val_errors = np.array(val_errors)
    q99 = np.percentile(val_errors, 99)
    print(f'   Threshold (Q99): {q99:.6f}')
    
    np.save(thr_path, np.array([q99]))
    print('\n[DONE] Model and threshold saved in models/')

if __name__ == '__main__':
    main()
