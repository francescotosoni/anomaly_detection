import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import glob
import os
import joblib

class CustomCategoryScaler:
    def __init__(self):
        # 6 joint positions (0-5), 6 currents (6-11), 6 velocities (12-17), 6 forces (18-23)
        self.categories = {
            'q': slice(0, 6),
            'current': slice(6, 12),
            'qd': slice(12, 18),
            'tcp': slice(18, 24)
        }
        self.means = {}
        self.stds = {}
        
    def fit(self, X):
        for cat, slc in self.categories.items():
            cat_data = X[:, slc]
            self.means[cat] = np.mean(cat_data)
            self.stds[cat] = np.std(cat_data)
            if self.stds[cat] == 0:
                self.stds[cat] = 1.0
        return self
                
    def transform(self, X):
        X_scaled = np.zeros_like(X)
        for cat, slc in self.categories.items():
            X_scaled[:, slc] = (X[:, slc] - self.means[cat]) / self.stds[cat]
        return X_scaled
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- TARGET DIRECTORIES ---
BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
DIR_NORMAL = os.path.join(BASE_DIR, "normal")

# Every anomaly category to analyse
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

# Select the features relevant to anomaly detection
FEATURES = [
    'actual_q_0', 'actual_q_1', 'actual_q_2',
    'actual_q_3', 'actual_q_4', 'actual_q_5',
    'actual_current_0', 'actual_current_1', 'actual_current_2', 
    'actual_current_3', 'actual_current_4', 'actual_current_5',
    'actual_qd_0', 'actual_qd_1', 'actual_qd_2', 
    'actual_qd_3', 'actual_qd_4', 'actual_qd_5',
    'actual_TCP_force_0', 'actual_TCP_force_1', 'actual_TCP_force_2',
    'actual_TCP_force_3', 'actual_TCP_force_4', 'actual_TCP_force_5'
]

WINDOW_SIZE = 50
STEP_SIZE = 1

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
    # Support both direct patterns and those inside inner 'test/' folders
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
        print("  -> Fitting the scaler on all healthy data...")
        concat_raw = np.vstack(all_raw_data)
        scaler.fit(concat_raw)
        # Save the scaler for the deployment / inference stage
        joblib.dump(scaler, "data/scaler.pkl")

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
    scaler = CustomCategoryScaler()
    X_normal = process_directory(DIR_NORMAL, scaler, is_training=True)
    np.save("data/X_normal.npy", X_normal)
    print(f"   Healthy windows shape: {X_normal.shape}")
    
    print("\n2. Processing all anomaly categories...")
    for anomaly in ANOMALY_TYPES:
        anom_dir = os.path.join(BASE_DIR, anomaly)
        X_anom = process_directory(anom_dir, scaler, is_training=False)
        if X_anom.size > 0:
            np.save(f"data/X_anomaly_{anomaly}.npy", X_anom)
            print(f"   [Saved] {anomaly}: {X_anom.shape}")
        else:
            print(f"   [Empty] No valid data for {anomaly}")
            
    print("\n[OK] Complete datasets extracted and normalised!")

if __name__ == "__main__":
    main()