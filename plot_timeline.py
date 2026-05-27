import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import glob
import os
from train_CNN import CNNAutoencoder
from train_LSTM import LSTMAutoencoder

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
TEST_ANOMALY = "push"  # Change this to 'drop_object', 'tcp_wrench', etc. to see other test cases
WINDOW_SIZE = 50
HIDDEN_DIM_LSTM = 32

FEATURES = [
    'actual_current_0', 'actual_current_1', 'actual_current_2', 
    'actual_current_3', 'actual_current_4', 'actual_current_5',
    'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 
    'actual_qd_3', 'actual_qd_4', 'actual_qd_5',
    'actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2',
    'actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5'
]

def load_first_csv(anomaly_type):
    folder = os.path.join(BASE_DIR, anomaly_type)
    # Search for CSVs in root folder or inside a "test" subfolder
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "test", "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found for anomaly type '{anomaly_type}'")
    return files[0]

def main():
    print(f"Loading timeline for {TEST_ANOMALY}...")
    csv_file = load_first_csv(TEST_ANOMALY)
    df = pd.read_csv(csv_file, sep=' ')
    df = df[FEATURES].dropna()
    raw_data = df.values
    
    # Load Scaler & Models
    scaler = joblib.load("scaler.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(FEATURES)
    
    # Init CNN
    cnn_model = CNNAutoencoder(n_features).to(device)
    cnn_model.load_state_dict(torch.load("cnn_autoencoder.pth", weights_only=True))
    cnn_model.eval()

    # Init LSTM
    lstm_model = LSTMAutoencoder(n_features, HIDDEN_DIM_LSTM).to(device)
    lstm_model.load_state_dict(torch.load("lstm_autoencoder.pth", weights_only=True))
    lstm_model.eval()

    # Fixed thresholds calculated during the evaluation phase
    threshold_cnn = 8.74
    threshold_lstm = 9.26

    # Scale Data
    scaled_data = scaler.transform(raw_data)
    
    # Create windows with STEP=1 for high-res timeline matching exactly what edge device will do
    windows = []
    for i in range(0, len(scaled_data) - WINDOW_SIZE + 1, 1):
        windows.append(scaled_data[i : i + WINDOW_SIZE])
    windows = np.array(windows)
    
    if len(windows) == 0:
        print("File is too short for a single window.")
        return

    # Compute MSE Errors
    cnn_tensor = torch.tensor(windows, dtype=torch.float32).transpose(1, 2).to(device)
    lstm_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        # CNN inference (Max error instead of mean)
        cnn_reconstruction = cnn_model(cnn_tensor)
        cnn_loss = criterion(cnn_reconstruction, cnn_tensor)
        cnn_errors = cnn_loss.mean(dim=2).max(dim=1)[0].cpu().numpy()
        
        # LSTM inference (Max error instead of mean)
        lstm_reconstruction = lstm_model(lstm_tensor)
        lstm_loss = criterion(lstm_reconstruction, lstm_tensor)
        lstm_errors = lstm_loss.mean(dim=1).max(dim=1)[0].cpu().numpy()

    # Time axis (aligning the error with the end of the window buffer to simulate when prediction actually occurs)
    time_axis_raw = np.arange(len(raw_data))
    time_axis_error = np.arange(WINDOW_SIZE - 1, len(raw_data))

    # --- PLOT TIMELINE ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Panel 1: Raw Physical Current
    ax1.plot(time_axis_raw, raw_data[:, 0], color='gray', label='Joint 0 Current (Base) Profile', alpha=1.0)
    ax1.set_title(f'Raw Telemetry Timeline - Scenario: {TEST_ANOMALY.capitalize()}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Current (A)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')

    # Panel 2: CNN Error
    ax2.plot(time_axis_error, cnn_errors, color='dodgerblue', label='CNN 1D Reconstruction Error (MSE)', linewidth=2)
    ax2.axhline(threshold_cnn, color='black', linestyle='--', linewidth=2, label=f'Alarm Threshold (MSE={threshold_cnn})')
    ax2.fill_between(time_axis_error, 0, cnn_errors, where=(cnn_errors > threshold_cnn), color='dodgerblue', alpha=0.4)
    ax2.set_title('Real-Time Edge Detection Simulation: CNN 1D', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MSE Error')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')

    # Panel 3: LSTM Error
    ax3.plot(time_axis_error, lstm_errors, color='darkorange', label='LSTM Reconstruction Error (MSE)', linewidth=2)
    ax3.axhline(threshold_lstm, color='black', linestyle='--', linewidth=2, label=f'Alarm Threshold (MSE={threshold_lstm})')
    ax3.fill_between(time_axis_error, 0, lstm_errors, where=(lstm_errors > threshold_lstm), color='darkorange', alpha=0.4)
    ax3.set_title('Real-Time Edge Detection Simulation: LSTM', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Steps (1 step = 10 ms)')
    ax3.set_ylabel('MSE Error')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')
    
    plt.xlim(0, len(raw_data))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()