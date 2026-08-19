import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from utils.feature_engineering import apply_feature_mask

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001

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

# --- MODEL DEFINITION (CNN 1D Autoencoder + BatchNorm + Dropout) ---
class CNNAutoencoder(nn.Module):
    def __init__(self, n_features):
        super(CNNAutoencoder, self).__init__()
        
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), 
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)  
        )
        
        # DECODER
        self.decoder = nn.Sequential(
            # 5 -> 25
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=5, stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 25 -> 50
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Refinement
            nn.Conv1d(in_channels=64, out_channels=n_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNNUpAutoencoder(nn.Module):
    """
    Variant of CNNAutoencoder whose decoder uses Upsample + Conv1d instead of
    ConvTranspose1d. Identical encoder.

    WHY: on ESP32-S3 the ConvTranspose1d CNN measures 644 ms per inference against
    an 80 ms budget. ESP-NN, Espressif's accelerated kernel library, implements no
    TransposeConv (only conv, depthwise conv, fully connected, pooling, add, mul,
    relu, softmax), so those two layers fall back to the TFLite Micro reference
    implementation. Conv1d is accelerated instead, and nearest-neighbour upsampling
    is a memory copy with no multiplications.

    The substitution is the one recommended in the literature regardless of speed:
    ConvTranspose produces checkerboard artefacts when the kernel is not a multiple
    of the stride. Here kernel and stride coincide (5/5 and 2/2), so that problem
    did not arise, but capacity stays comparable because the per-layer parameter
    count is the same for an equal kernel size.

    Spatial sizes are identical to the original: 5 -> 25 -> 50.

    WHY IT IS WRITTEN IN 2D WITH HEIGHT 1 rather than with 1D Conv1d/Upsample:
    the Edge Impulse ONNX->TFLite converter lifts 3D tensors to 4D and, on Resize,
    applies the scale factor to BOTH spatial axes. With the 1D version the ONNX was
    correct ([1,64,25] -> [1,64,50], last axis only), but the resulting TFLite held
    a 200 KB (1,64,50,64) tensor in place of a 3 KB (1,1,50,64) one: upsampling grew
    the channel axis too, inflating the arena (260 KB against 31) and the compute
    time. Writing kernel (1,k) and scale_factor (1,s) explicitly removes the
    ambiguity, and the graph becomes structurally identical to the original CNN,
    which converts correctly.

    Input arrives as (batch, channels, time) so the class stays interchangeable
    with CNNAutoencoder: the height axis is added and removed in here.
    """
    def __init__(self, n_features):
        super(CNNUpAutoencoder, self).__init__()

        # ENCODER - same topology as CNNAutoencoder, in (N, C, 1, L) form
        self.encoder = nn.Sequential(
            nn.Conv2d(n_features, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(64, 32, kernel_size=(1, 3), padding=(0, 2), dilation=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 5))
        )

        # DECODER - Upsample + Conv in place of ConvTranspose. The scale factor is
        # (1, s): the height axis must NOT grow.
        self.decoder = nn.Sequential(
            # 5 -> 25
            nn.Upsample(scale_factor=(1, 5), mode='nearest'),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 25 -> 50
            nn.Upsample(scale_factor=(1, 2), mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Refinement
            nn.Conv2d(64, n_features, kernel_size=(1, 3), padding=(0, 1))
        )

    def forward(self, x):
        x = x.unsqueeze(2)                  # (N, C, L) -> (N, C, 1, L)
        decoded = self.decoder(self.encoder(x))
        return decoded.squeeze(2)           # (N, C, 1, L) -> (N, C, L)


def main():
    print('1. Loading Data & Splitting (70% Train, 10% Val, 20% Test)...')
    X_normal_all = np.load('data/X_normal.npy')
    
    # SPLIT: 80% (Train+Val) and 20% (Test)
    X_temp, X_test_normal = train_test_split(X_normal_all, test_size=0.20, random_state=42)
    # SPLIT: 80% splits into 70% total (Train) and 10% total (Val) => 10/80 = 0.125
    X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)
    
    # Save the split Test Normal Set for the evaluation script (WITH ALL 24 FEATURES)
    np.save('data/X_test_normal.npy', X_test_normal)
    
    # APPLY FEATURE MASK FOR SEQUENTIAL MODELS
    X_train = apply_feature_mask(X_train, model_type='sequential')
    X_val = apply_feature_mask(X_val, model_type='sequential')
    
    n_features = X_train.shape[2]
    
    print(f'   Healthy Train: {X_train.shape[0]} | Healthy Val: {X_val.shape[0]} | Healthy Test: {X_test_normal.shape[0]}')
    
    # Rearranging dimensions for PyTorch Conv1d: (Batch, Features, Time)
    X_train_pt = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    X_val_pt = torch.tensor(X_val, dtype=torch.float32).transpose(1, 2)
    
    train_loader = DataLoader(TensorDataset(X_train_pt, X_train_pt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_pt, X_val_pt), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'   Using device: {device}')
    
    model = CNNAutoencoder(X_train_pt.shape[1]).to(device)
    
    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024
    print(f'   Model Parameters: {total_params:,} | Size: {model_size_kb:.1f} KB (float32)')
    
    criterion_train = nn.MSELoss()
    criterion_eval = nn.MSELoss(reduction='none')  # For calculating threshold per window
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    best_val_loss = float('inf')
    best_threshold = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15
    
    print('\n2. Starting Training (CNN 1D + BatchNorm + Dropout) -> q99 Threshold...')
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
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
        scheduler.step()
        
        # --- VALIDATION PHASE (And q99 Threshold Calculation) ---
        model.eval()
        val_loss = 0.0
        val_errors = []
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                
                # 1. Standard Val Loss (composite)
                loss = criterion_train(outputs, batch_x)
                val_loss += loss.item() * batch_x.size(0)
                
                # 2. Window-Level Reconstruction Error (MSE over Time -> Max over Features)
                raw_errors = criterion_eval(outputs, batch_x)
                window_errors = raw_errors.mean(dim=2).max(dim=1)[0].cpu().numpy()
                val_errors.extend(window_errors)
                
        val_loss /= len(val_loader.dataset)
        
        # Calculate q99 threshold for this epoch
        q99_threshold = np.percentile(val_errors, 99)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'   Epoch {epoch+1:03d}/{EPOCHS} -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | q99: {q99_threshold:.4f} | LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold = q99_threshold
            patience_counter = 0
            # Save Model and Threshold
            torch.save(model.state_dict(), 'models/cnn_autoencoder.pth')
            np.save('models/cnn_q99_threshold.npy', np.array([best_threshold]))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f'\n   [!] EARLY STOPPING triggered at epoch {epoch+1}. Stall limit reached.')
                break
        
    print(f'\n[V] CNN Training completed in {time.time() - start_time:.2f} seconds.')
    print(f'   Best Model saved with Val Loss: {best_val_loss:.8f}')
    print(f'   Optimal Deployment Threshold (q99): {best_threshold:.4f}')

if __name__ == '__main__':
    main()
