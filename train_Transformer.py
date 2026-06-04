import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
D_MODEL = 64
NHEAD = 2
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 128
BOTTLENECK_DIM = 32
DROPOUT = 0.1
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

# --- MODEL DEFINITION (Lightweight Transformer Autoencoder) ---
class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=2, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=128, bottleneck_dim=32,
                 seq_len=50, dropout=0.1):
        super(TransformerAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Input projection: n_features -> d_model
        self.input_proj = nn.Linear(n_features, d_model)
        
        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer Encoder (self-attention layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        
        # Bottleneck: mean pool across time -> compress -> expand -> repeat
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.ReLU()
        )
        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.ReLU()
        )
        
        # Transformer Decoder (self-attention layers on repeated bottleneck)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Output projection: d_model -> n_features
        self.output_proj = nn.Linear(d_model, n_features)
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        
        # Project to d_model space and add positional encoding
        x_proj = self.input_proj(x) + self.pos_encoding    # (batch, seq_len, d_model)
        
        # ENCODE: self-attention across time
        encoded = self.transformer_encoder(x_proj)          # (batch, seq_len, d_model)
        
        # BOTTLENECK: mean pool -> compress -> expand -> repeat
        pooled = encoded.mean(dim=1)                        # (batch, d_model)
        compressed = self.bottleneck(pooled)                 # (batch, bottleneck_dim)
        expanded = self.expand(compressed)                   # (batch, d_model)
        repeated = expanded.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, d_model)
        
        # Add positional encoding to decoder input
        repeated = repeated + self.pos_encoding
        
        # DECODE: self-attention for reconstruction
        decoded = self.transformer_decoder(repeated)         # (batch, seq_len, d_model)
        
        # Project back to feature space
        output = self.output_proj(decoded)                   # (batch, seq_len, n_features)
        return output

def main():
    print('1. Loading Data & Splitting (70% Train, 10% Val, 20% Test)...')
    X_normal = np.load('X_normal.npy')
    
    # SPLIT: 80% (Train+Val) and 20% (Test)
    X_temp, X_test_normal = train_test_split(X_normal, test_size=0.20, random_state=42)
    # SPLIT: 80% splits into 70% total (Train) and 10% total (Val) => 10/80 = 0.125
    X_train, X_val = train_test_split(X_temp, test_size=0.125, random_state=42)
    
    print(f'   Healthy Train: {X_train.shape[0]} | Healthy Val: {X_val.shape[0]} | Healthy Test: {X_test_normal.shape[0]}')
    
    data_tensor_train = torch.tensor(X_train, dtype=torch.float32)
    data_tensor_val = torch.tensor(X_val, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(data_tensor_train, data_tensor_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(data_tensor_val, data_tensor_val), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'   Using device: {device}')
    
    n_features = X_train.shape[2]
    model = TransformerAutoencoder(
        n_features, d_model=D_MODEL, nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS, num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, bottleneck_dim=BOTTLENECK_DIM,
        seq_len=SEQ_LEN, dropout=DROPOUT
    ).to(device)
    
    # Print model size info
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024
    print(f'   Model Parameters: {total_params:,} | Size: {model_size_kb:.1f} KB (float32)')
    
    criterion_train = CompositeLoss(alpha=0.8)
    criterion_eval = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    
    best_val_loss = float('inf')
    best_threshold = 0.0
    patience_counter = 0
    EARLY_STOP_PATIENCE = 15
    
    print('\n2. Starting Training (Lightweight Transformer Autoencoder) -> q99 Threshold...')
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
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_errors = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                
                loss = criterion_train(outputs, batch_x)
                val_loss += loss.item() * batch_x.size(0)
                
                # Error aggregation: mean over time (dim=1), max over features (dim=1)
                raw_errors = criterion_eval(outputs, batch_x)
                window_errors = raw_errors.mean(dim=1).max(dim=1)[0].cpu().numpy()
                val_errors.extend(window_errors)
                
        val_loss /= len(val_loader.dataset)
        
        q99_threshold = np.percentile(val_errors, 99)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'   Epoch {epoch+1:03d}/{EPOCHS} -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | q99: {q99_threshold:.4f} | LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_threshold = q99_threshold
            patience_counter = 0
            torch.save(model.state_dict(), 'transformer_autoencoder.pth')
            np.save('transformer_q99_threshold.npy', np.array([best_threshold]))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f'\n   [!] EARLY STOPPING triggered at epoch {epoch+1}. Stall limit reached.')
                break
                
    print(f'\n[V] Transformer Training completed in {time.time() - start_time:.2f} seconds.')
    print(f'   Best Model saved with Val Loss: {best_val_loss:.8f}')
    print(f'   Optimal Deployment Threshold (q99): {best_threshold:.4f}')

if __name__ == '__main__':
    main()
