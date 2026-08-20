import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_engineering import apply_feature_mask
from utils import window_dataset

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
HIDDEN_DIM = 32
N_LAYERS = 2
DROPOUT = 0.2
SEQ_LEN = 50

# --- COMPOSITE LOSS (MSE + Cosine Similarity) ---
class CompositeLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = nn.CosineSimilarity(dim=-1)
        self.alpha = alpha
    
    def forward(self, x, target):
        mse_loss = self.mse(x, target)
        cos_loss = (1 - self.cosine(
            x.reshape(x.size(0), -1), 
            target.reshape(target.size(0), -1)
        )).mean()
        return self.alpha * mse_loss + (1 - self.alpha) * cos_loss

# --- MODEL DEFINITION (LSTM Autoencoder with TRUE Bottleneck) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim, seq_len=50, n_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # Encoder: compresses entire sequence into last hidden state
        self.encoder = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_dim,
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        # Bridge: transforms bottleneck vector for decoder input
        self.bridge = nn.Linear(hidden_dim, hidden_dim)
        
        # Decoder: reconstructs sequence from repeated bottleneck vector
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim,
            num_layers=n_layers, 
            batch_first=True, 
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        # Output projection: maps decoder hidden state back to feature space
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x is (Batch, Time=seq_len, Features)
        
        # ENCODE: take ONLY the last hidden state -> TRUE bottleneck
        # Compresses (batch, 50, 18) = 900 values into (batch, 32) = 32 values
        _, (h_n, _) = self.encoder(x)              # h_n: (n_layers, batch, hidden_dim)
        bottleneck = torch.relu(self.bridge(h_n[-1]))  # (batch, hidden_dim)
        
        # DECODE: repeat bottleneck across time axis, then reconstruct
        repeated = bottleneck.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        decoded, _ = self.decoder(repeated)         # (batch, seq_len, hidden_dim)
        output = self.output_layer(decoded)         # (batch, seq_len, n_features)
        return output

def main():
    args = window_dataset.build_args('LSTM')
    sfx = args.suffix
    data_path = args.data or f'data/X_normal{sfx}.npy'
    model_path = f'models/lstm_autoencoder{sfx}.pth'
    thr_path = f'models/lstm{sfx}_q99_threshold.npy'

    print('1. Loading Data & Splitting (70% Train, 10% Val, 20% Test)...')
    print(f'   source: {data_path}')
    train_ds, val_ds, n_test = window_dataset.prepare(
        data_path, sfx, test_path=f'data/X_test_normal{sfx}.npy', seed=args.seed,
        model_type='sequential', channels_first=False)
    n_features = train_ds.n_features

    print(f'   Healthy Train: {len(train_ds)} | Healthy Val: {len(val_ds)} | Healthy Test: {n_test}')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'   Using device: {device}')
    
    model = LSTMAutoencoder(
        n_features, HIDDEN_DIM, 
        seq_len=SEQ_LEN, n_layers=N_LAYERS, dropout=DROPOUT
    ).to(device)
    
    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024
    print(f'   Model Parameters: {total_params:,} | Size: {model_size_kb:.1f} KB (float32)')
    
    criterion_train = nn.MSELoss()
    criterion_eval = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # ReduceLROnPlateau, matching train_MLP.py, rather than CosineAnnealingWarmRestarts.
    #
    # WHY: warm restarts put the validation minimum at the end of each cosine cycle,
    # and with T_0=15, T_mult=2 the cycle boundaries fall at epochs 15, 45, 105.
    # An early-stopping patience of 15 epochs therefore expires mid-cycle, before the
    # learning rate has annealed far enough to beat the previous cycle's minimum, so
    # the run can never complete the cycle after its best one. Measured on the
    # branch-inclusive dataset: the CNN peaked at epoch 14 (end of cycle 1) and stopped
    # at 29, never finishing cycle 2; the LSTM peaked at epoch 45 (end of cycle 2) and
    # stopped at 60, never finishing cycle 3. Both had a still-decreasing training loss
    # at the stop, i.e. they were truncated, not overfitting.
    #
    # Fixing the patience instead would need it longer than the longest cycle, and would
    # leave the three architectures on different schedules - which confounds an
    # architecture comparison with a training-recipe difference. A plateau scheduler has
    # no restarts, so patience and schedule cannot fight each other.
    #
    # The optimiser, weight decay and gradient clipping are left as they were: they are
    # per-architecture choices, not the cause of the truncation.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    best_threshold = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15
    
    print('\n2. Starting Training (LSTM Bottleneck Autoencoder) -> q99 Threshold...')
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion_train(outputs, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_errors = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                
                loss = criterion_train(outputs, batch_x)
                val_loss += loss.item() * batch_x.size(0)
                
                # raw_errors is (Batch, Time, Features)
                raw_errors = criterion_eval(outputs, batch_x)
                # Mean over Time (dim=1), Max over Features (dim=1 effectively after mean)
                window_errors = raw_errors.mean(dim=1).max(dim=1)[0].cpu().numpy()
                val_errors.extend(window_errors)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        q99_threshold = np.percentile(val_errors, 99)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'   Epoch {epoch+1:03d}/{EPOCHS} -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | q99: {q99_threshold:.4f} | LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold = q99_threshold
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            np.save(thr_path, np.array([best_threshold]))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f'\n   [!] EARLY STOPPING triggered at epoch {epoch+1}. Stall limit reached.')
                break
                
    print(f'\n[V] LSTM Training completed in {time.time() - start_time:.2f} seconds.')
    print(f'   Best Model saved with Val Loss: {best_val_loss:.8f}')
    print(f'   Optimal Deployment Threshold (q99): {best_threshold:.4f}')

if __name__ == '__main__':
    main()
