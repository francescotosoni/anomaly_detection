import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from train_CNN import CNNAutoencoder
from evaluate_metrics import FEATURES, BASE_DIR, WINDOW_SIZE, get_anomaly_files

# --- CONFIGURATION ---
CATEGORIES_TO_PLOT = ["force_stop", "drop_object", "change_weight"]
BATCH_SIZE = 512

def compute_errors_batched(scaled_data, model, device):
    """
    Batch inference: costruisce tutte le finestre in una volta sola e
    le processa in batch. Ordini di grandezza più veloce del loop singolo.
    """
    n = len(scaled_data)
    n_windows = n - WINDOW_SIZE + 1

    # Costruisce il tensore di tutte le finestre (shape: n_windows x WINDOW_SIZE x n_features)
    # usando stride_tricks per evitare copie in memoria
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled_data, (WINDOW_SIZE, scaled_data.shape[1])
    ).squeeze(1)   # (n_windows, WINDOW_SIZE, n_features)

    errors_windows = np.zeros(n_windows)
    criterion = nn.MSELoss(reduction='none')

    model.eval()
    with torch.no_grad():
        for start in range(0, n_windows, BATCH_SIZE):
            batch = torch.tensor(
                windows[start : start + BATCH_SIZE],
                dtype=torch.float32
            ).to(device)

            # CNN vuole (batch, n_features, seq_len)
            batch_t = batch.transpose(1, 2)
            recon = model(batch_t)
            loss = criterion(recon, batch_t)
            # max sul canale features, mean sul tempo → un errore scalare per finestra
            errs = loss.mean(dim=2).max(dim=1)[0].cpu().numpy()
            errors_windows[start : start + len(errs)] = errs

    # Allinea l'errore all'ULTIMO campione della finestra (per plot nel dominio temporale)
    errors_full = np.zeros(n)
    errors_full[WINDOW_SIZE - 1:] = errors_windows
    return errors_full


def get_mismatch_zone(y_true, preds):
    """
    Restituisce gli indici in cui la label è 1 ma il modello predice 0
    (zona di falsi negativi causati dal labeling persistente).
    """
    y_true = np.array(y_true)
    preds  = np.array(preds)
    # FN dove il segnale è già tornato normale ma la label persiste
    return (y_true == 1) & (preds == 0)


def plot_mismatch_for_category(category, model, scaler, device, threshold, ax_row, show_xlabel):
    """Genera i 3 subplot per una singola categoria."""
    files = get_anomaly_files(category, max_files=1)
    if not files:
        for ax in ax_row:
            ax.set_visible(False)
        return

    test_file = files[0]
    print(f"  [{category}] -> {os.path.basename(test_file)}")

    df = pd.read_csv(test_file, sep=' ').dropna(subset=FEATURES)

    # 1. Label hardware reale (bit digitale)
    bits = df['actual_digital_input_bits'].values if 'actual_digital_input_bits' in df.columns \
           else np.zeros(len(df), dtype=int)
    y_true = np.zeros(len(bits), dtype=int)
    anom_indices = np.where(bits == 32)[0]
    if len(anom_indices) > 0:
        # Push/tcp_wrench: label solo mentre il flag è attivo
        if category in ("push", "tcp_wrench"):
            y_true[bits == 32] = 1
        else:
            # Tutte le altre: label persiste dalla prima anomalia → fine file (il mismatch!)
            y_true[anom_indices[0]:] = 1

    # 2. Batch inference
    scaled_data = scaler.transform(df[FEATURES].values)
    errors = compute_errors_batched(scaled_data, model, device)
    preds  = (errors > threshold).astype(int)

    # 3. Zona di mismatch: label=1 ma modello dice 0
    mismatch_mask = get_mismatch_zone(y_true, preds)

    t = np.arange(len(y_true))

    ax1, ax2, ax3 = ax_row

    # ── Subplot 1: Ground Truth Label ──────────────────────────────────────────
    ax1.fill_between(t, 0, y_true, color='#e74c3c', alpha=0.35, label='Label dataset (hardware flag)')
    ax1.plot(t, y_true, color='#e74c3c', linewidth=1.2)
    # Evidenzia la zona di mismatch in arancione
    ax1.fill_between(t, 0, mismatch_mask.astype(int),
                     color='#f39c12', alpha=0.55, label='⚠ Mismatch Area (physically normal)')
    ax1.set_ylabel(f"{category}\nLabel", fontsize=9)
    ax1.set_ylim(-0.05, 1.35)
    ax1.set_yticks([0, 1])
    ax1.legend(loc='upper right', fontsize=7.5)
    ax1.grid(axis='x', alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # ── Subplot 2: Errore di ricostruzione ────────────────────────────────────
    ax2.plot(t, errors, color='#2980b9', linewidth=0.8, label='CNN Reconstruction Error')
    ax2.axhline(threshold, color='#2c3e50', linestyle='--', linewidth=1.5,
                label=f'q99 Threshold ({threshold:.2f})')
    # Ombreggiatura mismatch anche sull'errore
    ax2.fill_between(t, 0, errors, where=mismatch_mask,
                     color='#f39c12', alpha=0.4, label='⚠ Mismatch Area')
    ax2.set_ylabel('MSE Error', fontsize=9)
    ax2.legend(loc='upper right', fontsize=7.5)
    ax2.grid(axis='x', alpha=0.3)
    ax2.tick_params(labelbottom=False)

    # ── Subplot 3: Predizioni modello ─────────────────────────────────────────
    ax3.fill_between(t, 0, preds, color='#27ae60', alpha=0.4, label='Model Prediction')
    ax3.plot(t, preds, color='#27ae60', linewidth=1.0)
    # Zona mismatch: dove il modello dice 0 ma label dice 1
    ax3.fill_between(t, 0, mismatch_mask.astype(int),
                     color='#f39c12', alpha=0.55, label='⚠ FN (persistent label)')
    ax3.set_ylabel('Pred. Label', fontsize=9)
    ax3.set_ylim(-0.05, 1.35)
    ax3.set_yticks([0, 1])
    ax3.legend(loc='upper right', fontsize=7.5)
    ax3.grid(axis='x', alpha=0.3)

    if show_xlabel:
        ax3.set_xlabel('Time (samples)', fontsize=9)

    # Annotazione testuale sulla zona mismatch
    first_mm = np.where(mismatch_mask)[0]
    if len(first_mm) > 0:
        mid = first_mm[len(first_mm) // 2]
        ax3.annotate(
            'Persistent label\nbut signal\nis normal',
            xy=(mid, 0.5), xytext=(mid - len(t) * 0.18, 0.75),
            fontsize=7.5, color='#e67e22', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.3)
        )


def main():
    print("Generating Mismatch Visualizations (one per category)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler    = joblib.load("scaler.pkl")
    threshold = np.load('cnn_q99_threshold.npy')[0]

    model = CNNAutoencoder(len(FEATURES)).to(device)
    model.load_state_dict(torch.load("cnn_autoencoder.pth", map_location=device, weights_only=True))
    model.eval()

    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', alpha=0.5, label='Label dataset = 1 (declared anomaly)'),
        mpatches.Patch(facecolor='#27ae60', alpha=0.5, label='Model Prediction = 1 (anomaly detected)'),
        mpatches.Patch(facecolor='#f39c12', alpha=0.7, label='⚠ Mismatch: label=1, model=0 (signal returned normal)'),
    ]

    for cat in CATEGORIES_TO_PLOT:
        # Una figura separata per categoria: 3 subplot verticali ben spaziati
        fig, axes = plt.subplots(
            nrows=3, ncols=1,
            figsize=(14, 13),
            gridspec_kw={'hspace': 0.12}
        )

        fig.suptitle(
            f'Label Mismatch — Category: "{cat}"\n'
            'Orange area = Label=1 but signal is physically normal (false negative from labeling)',
            fontsize=13, fontweight='bold'
        )

        fig.legend(handles=legend_elements, loc='upper right', fontsize=9,
                   bbox_to_anchor=(0.99, 0.94), framealpha=0.92)

        plot_mismatch_for_category(cat, model, scaler, device, threshold,
                                   ax_row=axes, show_xlabel=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = f"mismatch_{cat}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {plot_path}")
        plt.close(fig)

    print("\nDone. Generated:", [f"mismatch_{c}.png" for c in CATEGORIES_TO_PLOT])


if __name__ == "__main__":
    main()
