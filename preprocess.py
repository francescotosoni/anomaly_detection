import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import glob
import os
import joblib

# --- TARGET DIRECTORIES ---
BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
DIR_NORMAL = os.path.join(BASE_DIR, "normal")

# Tutte le categorie di anomalia da analizzare
ANOMALY_TYPES = [
    "change_weight", 
    "drop_object", 
    "force_stop", 
    "long_wait",
    "outside_box", 
    "push", 
    "scale_speed", 
    "tcp_wrench"
]

# Selezioniamo le features rilevanti per l'Anomaly Detection
FEATURES = [
    'actual_current_0', 'actual_current_1', 'actual_current_2', 
    'actual_current_3', 'actual_current_4', 'actual_current_5',
    'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 
    'actual_qd_3', 'actual_qd_4', 'actual_qd_5',
    'actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2',
    'actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5'
]

WINDOW_SIZE = 50
STEP_SIZE = 10

def load_and_filter_data(filepath, sep=' '):
    df = pd.read_csv(filepath, sep=sep)
    df = df[FEATURES].dropna()
    return df.values

def create_sliding_windows(data, window_size, step_size):
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i : i + window_size])
    if len(windows) == 0:
        return np.array([])
    return np.array(windows)

def process_directory(directory_path, scaler, is_training=False):
    # Supportiamo sia i pattern diretti che quelli in cartelle "test/" interne
    csv_files = glob.glob(os.path.join(directory_path, "*.csv")) + glob.glob(os.path.join(directory_path, "test", "*.csv"))
    print(f"  -> Trovati {len(csv_files)} file in {os.path.basename(directory_path)}")
    
    all_raw_data = []
    for file in csv_files:
        data = load_and_filter_data(file)
        if len(data) >= WINDOW_SIZE:
            all_raw_data.append(data)
            
    if not all_raw_data:
        return np.array([])

    if is_training:
        print("  -> Fittaggio dello scaler su tutti i dati sani...")
        concat_raw = np.vstack(all_raw_data)
        scaler.fit(concat_raw)
        # Salva lo scaler per la fase di deployment / inferenza
        joblib.dump(scaler, "scaler.pkl")

    all_windows = []
    for data in all_raw_data:
        scaled_data = scaler.transform(data)
        windows = create_sliding_windows(scaled_data, WINDOW_SIZE, STEP_SIZE)
        if windows.size > 0:
            all_windows.append(windows)

    if all_windows:
        return np.vstack(all_windows)
    return np.array([])

def main():
    print("1. Elaborazione della Baseline Globale (stochastic_PnP/normal)...")
    scaler = RobustScaler()
    X_normal = process_directory(DIR_NORMAL, scaler, is_training=True)
    np.save("X_normal.npy", X_normal)
    print(f"   Shape Finestre Sane: {X_normal.shape}")
    
    print("\n2. Elaborazione di Tutte le Categorie di Anomalia...")
    for anomaly in ANOMALY_TYPES:
        anom_dir = os.path.join(BASE_DIR, anomaly)
        X_anom = process_directory(anom_dir, scaler, is_training=False)
        if X_anom.size > 0:
            np.save(f"X_anomaly_{anomaly}.npy", X_anom)
            print(f"   [Salvato] {anomaly}: {X_anom.shape}")
        else:
            print(f"   [Vuoto] Nessun dato valido per {anomaly}")
            
    print("\n✅ Dataset completi estratti e normalizzati!")

if __name__ == "__main__":
    main()