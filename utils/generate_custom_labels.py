import os
import glob
import pandas as pd
import numpy as np
import json

BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
ANOMALY_CLASSES = [
    "push", "tcp_wrench", "drop_object", "change_weight", 
    "force_stop", "outside_box", "scale_speed"
]
SAMPLING_RATE = 100

def get_anomaly_files(category):
    folder = os.path.join(BASE_DIR, category)
    files = glob.glob(os.path.join(folder, "*.csv")) + glob.glob(os.path.join(folder, "test", "*.csv"))
    return files

def main():
    print("Generating custom labels for all anomalous files...")
    custom_labels = {}
    
    for cat in ANOMALY_CLASSES:
        files = get_anomaly_files(cat)
        for f in files:
            try:
                df = pd.read_csv(f, sep=' ')
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
                
            if 'actual_digital_input_bits' not in df.columns:
                continue
                
            bits = df['actual_digital_input_bits'].values
            anom_indices = np.where(bits == 32)[0]
            
            if len(anom_indices) == 0:
                continue
                
            start_idx = int(anom_indices[0])
            end_idx = len(df)
            
            # --- HEURISTICS ---
            if cat in ["push", "tcp_wrench"]:
                # Intermittent: End is the last trigger index
                end_idx = int(anom_indices[-1])
                
            elif cat in ["change_weight", "scale_speed", "outside_box"]:
                # Persistent: Anomaly lasts until the end of the file
                end_idx = len(df)
                
            elif cat == "force_stop":
                # Transient: Robot stops. End is when actual_qd goes effectively to zero.
                qd_cols = [f'actual_qd_{i}' for i in range(6)]
                if all(c in df.columns for c in qd_cols):
                    qd_data = df[qd_cols].values
                    # Search after start_idx for the first time velocity is near zero
                    for i in range(start_idx, len(df)):
                        max_vel = np.max(np.abs(qd_data[i]))
                        if max_vel < 0.01:  # 0.01 rad/s is practically stopped
                            # Add a small buffer of 0.5s (50 timesteps) to capture the final vibration
                            end_idx = min(i + 50, len(df))
                            break
                            
            elif cat == "drop_object":
                # Transient: High perturbation followed by normal behavior.
                # Use empirical 3.3s (330 timesteps) as a baseline guess for manual refinement.
                end_idx = min(start_idx + 330, len(df))
            
            # Save the relative path as key to make it robust
            rel_path = os.path.relpath(f, BASE_DIR).replace("\\", "/")
            custom_labels[rel_path] = [start_idx, end_idx]
            
    output_path = r"C:\Users\toso9\ur5_data_collection\data\custom_labels.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(custom_labels, f, indent=4)
        
    print(f"Generated labels for {len(custom_labels)} files.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
