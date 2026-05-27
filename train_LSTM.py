import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from sklearn.model_selection import train_test_split

# --- CONFIGURAZIONE ---
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_DIM = 32

# --- DEFINIZIONE MODELLO (LSTM Autoencoder) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # ENCODER: Ritorna lo stato nascosto finale
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, batch_first=True)
        
        # DECODER: Prende lo stato nascosto e lo espande temporalmente
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        # Livello lineare per mappare dall'hidden_dim al numero di features originali
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x.shape = (Batch, 50, Features)
        seq_len = x.shape[1]
        
        # Encode
        _, (hidden, _) = self.encoder(x) # hidden.shape = (1, Batch, HiddenDim)
        hidden = hidden[-1] # Prendi l'ultimo layer -> (Batch, HiddenDim)
        
        # Per decodificare, replichiamo l'hidden state per la lunghezza della sequenza
        # decoded_input.shape = (Batch, 50, HiddenDim)
        decoded_input = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoder_out, _ = self.decoder(decoded_input)
        
        # Mappa alle features
        reconstruction = self.output_layer(decoder_out)
        return reconstruction

def main():
    print("1. Caricamento Dati...")
    X_normal = np.load("X_normal.npy")
    X_anomaly = np.load("X_anomaly.npy")
    
    # SPLIT DEI DATI
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
    print(f"   Train Sani: {X_train.shape[0]} | Validation Sani: {X_val.shape[0]} | Test Anomali: {X_anomaly.shape[0]}")
    
    # LSTM aspetta (Batch, Time, Features) -> questo è già il nostro formato di default originario
    X_train_pt = torch.tensor(X_train, dtype=torch.float32)
    X_val_pt = torch.tensor(X_val, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_pt, X_train_pt), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_pt, X_val_pt), batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Utilizzando device: {device}")
    
    n_features = X_train_pt.shape[2]
    model = LSTMAutoencoder(n_features=n_features, hidden_dim=HIDDEN_DIM).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Inizializzo LR Scheduler: dimezza la forza d'apprendimento se non ci sono miglioramenti
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    
    # Inizializzo Early Stopping
    best_val_loss = float('inf')
    patience_counter = 0
    EARLY_STOP_PATIENCE = 10
    
    print("\n2. Inizio Addestramento (LSTM) con LR Scheduler ed Early Stopping...")
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
        
        # Passiamo la regression loss allo scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        print(f"   Epoca {epoch+1:03d}/{EPOCHS} -> Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.6f}")
        
        # Logica Early Stopping e salvataggio
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "lstm_autoencoder.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n   [!] EARLY STOPPING scattato all'epoca {epoch+1}. Raggiunto il limite di stallo.")
                break
        
    print(f"\n✅ Addestramento LSTM completato in {time.time() - start_time:.2f} secondi.")
    print(f"   Il Modello Perfetto è stato salvato con una Val Loss minima di: {best_val_loss:.8f}")

if __name__ == "__main__":
    main()