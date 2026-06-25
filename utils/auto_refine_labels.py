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
    print("Auto-Refining custom labels using physical kinematics...")
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
            
            # Extract basic kinematic features
            qd_cols = [f'actual_qd_{i}' for i in range(6)]
            force_cols = [f'actual_TCP_force_{i}' for i in range(6)]
            
            # --- PHYSICS-BASED HEURISTICS ---
            if cat in ["push", "tcp_wrench"]:
                # Intermittent: strictly bound to the physical button press
                start_idx = int(anom_indices[0])
                end_idx = int(anom_indices[-1])
                
            elif cat in ["change_weight", "scale_speed", "outside_box"]:
                # Persistent: Anomaly lasts until the end of the file
                start_idx = int(anom_indices[0])
                end_idx = len(df)
                
            elif cat == "force_stop":
                # Transient: Robot stops violently. 
                # Start: The true start is the moment of massive deceleration.
                # We search for it in a tight window around the hardware trigger.
                trigger_idx = int(anom_indices[0])
                start_idx = trigger_idx
                
                if all(c in df.columns for c in qd_cols):
                    qd_data = df[qd_cols].values
                    
                    acc = np.sum(np.abs(np.diff(qd_data, axis=0)), axis=1)
                    search_start = max(0, trigger_idx - 50)
                    search_end = min(len(acc), trigger_idx + 100)
                    
                    if search_start < search_end:
                        idx_spike_relative = int(np.argmax(acc[search_start:search_end]))
                        idx_spike = search_start + idx_spike_relative
                        
                        # Only use the spike if it's a real deceleration (e.g. > 0.05)
                        if acc[idx_spike] > 0.05:
                            start_idx = idx_spike
                    
                    # End: when absolute velocity of ALL joints is practically zero for 50 timesteps.
                    # If the robot is already completely stopped at start_idx, the loop will 
                    # immediately close the label, resulting in a 0-length anomaly (which is physically correct!).
                    consecutive_zero = 0
                    for i in range(start_idx, len(df)):
                        max_vel = np.max(np.abs(qd_data[i]))
                        if max_vel < 0.005:  # 0.005 rad/s is stationary
                            consecutive_zero += 1
                        else:
                            consecutive_zero = 0
                            
                        if consecutive_zero >= 50:
                            end_idx = i - 50
                            break
                            
            elif cat == "drop_object":
                # Transient: Object falls. 
                # Start: The hardware trigger (orange label) is actually perfectly synchronized with the drop event (when force returns to 0).
                # My previous heuristic was mistakenly confusing the violent "pick up" phase with the drop!
                start_idx = int(anom_indices[0])
                
                # End: The anomaly lasts during the vibration and the subsequent wrong-weight movement.
                # However, if the robot completely stops (flatline), the Autoencoder will not see an anomaly.
                # So we end it when the robot stops, OR after an empirical maximum of 3.3s (330 timesteps).
                end_idx = min(start_idx + 330, len(df))
                
                if all(c in df.columns for c in qd_cols):
                    qd_data = df[qd_cols].values
                    consecutive_zero = 0
                    for i in range(start_idx, len(df)):
                        max_vel = np.max(np.abs(qd_data[i]))
                        if max_vel < 0.005:
                            consecutive_zero += 1
                        else:
                            consecutive_zero = 0
                            
                        if consecutive_zero >= 50:
                            # Robot stopped. Cut the label early to avoid the flatline.
                            possible_end = i - 50
                            if possible_end > start_idx:
                                end_idx = min(end_idx, possible_end)
                            break
            
            # Save the relative path as key
            rel_path = os.path.relpath(f, BASE_DIR).replace("\\", "/")
            custom_labels[rel_path] = [start_idx, end_idx]
            
    output_path = r"C:\Users\toso9\ur5_data_collection\data\custom_labels.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(custom_labels, f, indent=4)
        
    print(f"Refined physical labels for {len(custom_labels)} files.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
