import os
import pandas as pd
import json
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\toso9\ur5_data_collection\data\data\stochastic_PnP"
LABELS_FILE = r"C:\Users\toso9\ur5_data_collection\data\custom_labels.json"

def main():
    if not os.path.exists(LABELS_FILE):
        print(f"File {LABELS_FILE} not found. Run generate_custom_labels.py first.")
        return
        
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        custom_labels = json.load(f)
        
    print(f"Loaded {len(custom_labels)} labels for verification.")
    print("Close the plot window to see the next file. Press Ctrl+C in terminal to abort.")
    
    for rel_path, (start_idx, end_idx) in custom_labels.items():
        # Handle cross-platform path separators
        file_path = os.path.join(BASE_DIR, rel_path.replace("/", os.sep))
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        try:
            df = pd.read_csv(file_path, sep=' ')
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Verification: {rel_path}\nAnomaly Bounds: [{start_idx}, {end_idx}]")
        
        # Plot Velocities
        qd_cols = [f'actual_qd_{i}' for i in range(6)]
        if all(c in df.columns for c in qd_cols):
            ax1.plot(df[qd_cols].values, alpha=0.7)
        ax1.set_title("Joint Velocities (actual_qd)")
        ax1.set_ylabel("rad/s")
        # Original labels
        if 'actual_digital_input_bits' in df.columns:
            bits = df['actual_digital_input_bits'].values
            import numpy as np
            anom_indices = np.where(bits == 32)[0]
            if len(anom_indices) > 0:
                first_idx = anom_indices[0]
                if "push" in rel_path or "tcp_wrench" in rel_path:
                    # Intermittent original label: multiple segments where bits == 32
                    # Approximate for visualization by showing from first to last
                    orig_end = anom_indices[-1]
                else:
                    # Persistent original label: to the end of the file
                    orig_end = len(df)
                    
                ax1.axvspan(first_idx, orig_end, color='orange', alpha=0.15, label='Original Label')
                ax2.axvspan(first_idx, orig_end, color='orange', alpha=0.15)
        
        ax1.axvspan(start_idx, end_idx, color='red', alpha=0.3, label='Custom Label')
        ax1.legend(loc='upper right')
        
        # Plot TCP Force
        force_cols = [f'actual_TCP_force_{i}' for i in range(6)]
        if all(c in df.columns for c in force_cols):
            ax2.plot(df[force_cols].values, alpha=0.7)
        ax2.set_title("TCP Force")
        ax2.set_ylabel("N / Nm")
        ax2.axvspan(start_idx, end_idx, color='red', alpha=0.3)
        ax2.set_xlabel("Timestep (10ms)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
