import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# --- MODEL DEFINITION (CNN 1D Autoencoder) ---
class CNNAutoencoder(nn.Module):
    def __init__(self, n_features):
        super(CNNAutoencoder, self).__init__()
        # PyTorch Conv1d expects input (Batch, Features, SequenceLength)
        
        # ENCODER (Adding Dilated Convolutions e più canali)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Drops from 50 to 25
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5)  # Drops from 25 to 5
        )
        
        # DECODER
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=5), # Rises from 5 to 25
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2), # Rises from 25 to 50
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=n_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def main():
    print("1. Loading Data...")
    X_normal = np.load("X_normal.npy")
    X_anomaly = np.load("X_anomaly.npy")
    
    # DATA SPLIT
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
    print(f"   Healthy Train: {X_train.shape[0]} | Healthy Validation: {X_val.shape[0]} | Anomalous Test: {X_anomaly.shape[0]}")
    
    # Rearranging dimensions per PyTorch: da (Batch, Time, Features) a (Batch, Features, Time)
    X_train_pt = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
    X_val_pt = torch.tensor(X_val, dtype=torch.float32).transpose(1, 2)
    
    train_loader = DataLoader(TensorDataset(X_train_pt, X_train_pt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_pt, X_val_pt), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    n_features = X_train_pt.shape[1]
    model = CNNAutoencoder(n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize LR Scheduler: dimezza la "forza" d'apprendimento se non ci sono miglioramenti
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    # Initialize Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10
    
    print("\n2. Inizio Addestramento (CNN 1D) con LR Scheduler ed Early Stopping...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_x)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        # Pass validation loss allo scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        print(f"   Epoca {epoch+1:03d}/{EPOCHS} -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # Early Stopping logic e salvataggio pesi perfetti
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "cnn_autoencoder.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n   [!] EARLY STOPPING triggered all'epoca {epoch+1}. Stall limit reached di stallo.")
                break
        
    print(f"\n✅ CNN Training completed in {time.time() - start_time:.2f} secondi.")
    print(f"   The Perfect Model è stato salvato con una Val Loss minima di: {best_val_loss:.8f}")

if __name__ == "__main__":
    main()