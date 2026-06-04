# Collaborative Robot (UR5) Time-Series Anomaly Detection

This repository contains an advanced **Time-Series Anomaly Detection (TSAD)** pipeline for Universal Robots UR5 collaborative manipulators. The project evaluates and compares four deep learning architectures — 1D CNN, LSTM, MLP, and Transformer — using unsupervised autoencoder-based learning, targeting both academic benchmarks and **physical MCU deployment via TinyML**.

---

## 📂 Repository Structure

```
ur5_data_collection/
│
├── preprocess.py            # Data ingestion, feature extraction, RobustScaler, windowing
├── collect.py               # Live data collection via RTDE from the UR5 robot
├── check_data.py            # Dataset validation and integrity checks
│
├── train_CNN.py             # 1D CNN Autoencoder (BatchNorm + Dilated Conv + Dropout)
├── train_LSTM.py            # LSTM Autoencoder (true bottleneck via RepeatVector)
├── train_MLP.py             # MLP Autoencoder (fully-connected, compact bottleneck)
├── train_Transformer.py     # Lightweight Transformer Autoencoder (mean-pool bottleneck)
│
├── evaluate_metrics.py      # Full evaluation: RAW / MCU Deploy / Point-Adjusted + per-category
├── plot_mismatch.py         # Visualization of the label mismatch problem
│
├── anomaly_config.xml       # Robot anomaly configuration
└── data/                    # Dataset (stochastic_PnP trajectories)
```

---

## 🧠 Approach: Unsupervised Autoencoder-Based Detection

All four models are trained as **Autoencoders** on purely *healthy (normal)* UR5 kinematic data. During training the model learns to compress and reconstruct normal movement patterns. At inference time, anomalous inputs (impacts, unexpected forces, weight changes, etc.) cannot be reconstructed accurately, causing the **Reconstruction Error** to spike above a calibrated threshold.

This unsupervised approach requires **no anomaly labels during training**, making it deployable on new robot setups without re-labeling.

**Input features (18 total):**
- Joint currents: `actual_current_[0-5]`
- Joint velocities: `actual_qd_[0-5]`
- TCP forces: `actual_TCP_force_[0-5]`

**Windowing:** `WINDOW_SIZE = 50` samples, `STEP_SIZE = 1` (overlapping, millisecond-level resolution)

---

## 🏗️ Model Architectures

### 1D CNN Autoencoder (`train_CNN.py`) — 44,658 params / 174 KB

```
Encoder:
  Conv1d(18→64, k=3)     → BatchNorm1d → ReLU → Dropout(0.1)
  Conv1d(64→64, k=3, dilation=2) → BatchNorm1d → ReLU → MaxPool(2)
  Conv1d(64→32, k=3, dilation=2) → BatchNorm1d → ReLU → MaxPool(5)

Decoder:
  Upsample(×5) → Conv1d(32→64, dilation=2) → BatchNorm1d → ReLU
  Upsample(×2) → Conv1d(64→64) → BatchNorm1d → ReLU
  Conv1d(64→18)
```

Dilated convolutions capture multi-scale temporal patterns without increasing parameter count. BatchNorm stabilises training; Dropout prevents overfitting on the normal distribution.

---

### LSTM Autoencoder (`train_LSTM.py`) — 33,650 params / 131 KB

```
Encoder: LSTM(18→32, layers=2, dropout=0.2)
         → only h_n[-1]: (batch, 32)   ← true bottleneck (900→32 values)
Bridge:  Linear(32→32) + ReLU
Repeat:  unsqueeze(1).repeat(seq_len)  ← RepeatVector pattern
Decoder: LSTM(32→32, layers=2, dropout=0.2)
Output:  Linear(32→18)
```

> **Key fix from original implementation:** the original LSTM passed the full encoded sequence `(batch, 50, 32)` to the decoder, making it an identity function (no compression → no anomaly sensitivity, Recall = 1%). The rewrite extracts only the last hidden state `h_n[-1]`, achieving a genuine 900→32 bottleneck. Result: RAW Recall went from **1% → 22%**, PA ROC-AUC from **0.824 → 0.969**.

---

### MLP Autoencoder (`train_MLP.py`) — 500,516 params / 1,955 KB

```
Input: (batch, 50, 18) → flatten → (batch, 900)
Encoder: 900 → 256 → 128 → [32]   (bottleneck)
Decoder: [32] → 128 → 256 → 900
Output: reshape → (batch, 50, 18)
BatchNorm + ReLU + Dropout(0.1) at each layer
```

Competitive performance but too large (1.9 MB) for TinyML entry-level MCUs. Included for academic comparison.

---

### Transformer Autoencoder (`train_Transformer.py`) — 143,666 params / 561 KB

```
Input (18) → Linear(64) + Learnable PosEncoding(50×64)
           → TransformerEncoder [2 layers, 2 heads, ff=128, dropout=0.1]
           → MeanPool over time → (batch, 64)
           → Linear(64→32) + ReLU          ← bottleneck
           → Linear(32→64) + ReLU + Repeat ← expand
           → TransformerDecoder [2 layers, 2 heads, ff=128, dropout=0.1]
           → Linear(64→18)
```

Self-attention allows the model to capture global temporal dependencies (O(N²) complexity). Highest per-category recall; best choice for safety-critical applications.

---

## ⚙️ Training Pipeline (Shared Across All Models)

All four models use a standardised training pipeline:

| Component | Configuration |
|-----------|---------------|
| **Loss function** | `CompositeLoss`: 80% MSE + 20% Cosine Similarity |
| **Optimiser** | Adam, lr=0.001 |
| **LR Scheduler** | `CosineAnnealingWarmRestarts` (T_0=15, T_mult=2) |
| **Gradient clipping** | `max_norm=1.0` |
| **Early stopping** | Patience = 15 epochs |
| **Max epochs** | 150 |
| **Batch size** | 64 |
| **Data split** | 70% Train / 10% Val / 20% Test (random_state=42) |
| **Threshold** | q99 percentile of validation reconstruction errors |

The CompositeLoss combines MSE (for absolute magnitude) with Cosine Similarity (for shape/direction), making the model sensitive to both the scale and the trajectory of reconstruction errors.

---

## 📊 Evaluation Framework (`evaluate_metrics.py`)

Three evaluation paradigms are reported for each model:

### [1] RAW Metrics — strict window-by-window
Each window is classified independently. Penalised by **label mismatch** (see below). Represents the lower-bound of real-world performance.

### [2] MCU Deploy Metrics — Alarm Persistence *(most realistic)*
Simulates the actual MCU deployment behaviour: once a window exceeds the threshold, the alarm remains active for **200 subsequent windows (= 2 seconds at 100 Hz)**. This reflects how an embedded system would behave in practice.

### [3] Point-Adjusted (PA) — Academic Benchmark
Standard in TSAD literature: if at least one window in a continuous anomalous segment is correctly flagged, the entire segment counts as detected. Represents the upper-bound; does not reflect real MCU behaviour.

### Threshold Sweep
Automatically evaluates q90, q95, q99 thresholds (computed on the test normal set) with and without alarm persistence, to find the optimal precision/recall trade-off.

### Per-Category Report
Recall with alarm persistence broken down for each of the 7 anomaly categories.

---

## 🔬 Key Technical Findings

### The Label Mismatch Problem

The dataset hardware flag (`actual_digital_input_bits == 32`) is set at the moment the anomaly begins and **persists until the end of the recording**, even after the robot has returned to normal behaviour. This causes RAW metrics to underestimate model performance for transient anomalies:

```
Time:   [--- normal ---][IMPACT][--- robot stabilised ---]
Label:  [0 0 0 0 0 0 0][   1  ][ 1  1  1  1  1  1  1  1 ] ← label persists
Model:  [0 0 0 0 0 0 0][   1  ][ 0  0  0  0  0  0  0  0 ] ← correctly "normal"
RAW:                      TP      FN FN FN FN FN FN FN FN ← penalised!
```

The `plot_mismatch.py` script generates visualisations of this phenomenon for `force_stop`, `drop_object`, and `change_weight`. `tcp_wrench` (continuous force application) reaches ~98% recall across all models, confirming the models work correctly where the anomaly is genuinely sustained.

### Threshold Sweep Results
The q99 threshold calibrated on the validation set is consistent with the q90 threshold on the test set. However, the large spread between q90 and q99 on the test set (e.g. LSTM: 8.81 → 446.73) suggests the presence of transition windows (robot start/stop, trajectory changes) in the test set that inflate the upper percentiles. Cleaning these would further stabilise thresholds.

---

## 📈 Results Summary

### MCU Deploy Metrics — Alarm Persistence 2s *(recommended for deployment)*

| Model | Size (KB) | Precision | Recall | **F1** |
|-------|-----------|-----------|--------|--------|
| CNN | 174 | 0.4118 | 0.4299 | 0.4207 |
| **LSTM** | **131** | 0.4088 | 0.5702 | **0.4762** 🥇 |
| MLP | 1,955 | 0.3614 | 0.5509 | 0.4364 |
| Transformer | 561 | 0.3677 | **0.5924** | 0.4537 |

### Point-Adjusted — Academic Benchmark

| Model | Precision | Recall | **F1** | ROC-AUC |
|-------|-----------|--------|--------|---------|
| **CNN** | **0.8638** | 0.9684 | **0.9131** 🥇 | 0.9544 |
| LSTM | 0.8353 | 0.9776 | 0.9008 | **0.9686** 🥇 |
| MLP | 0.8303 | 0.9797 | 0.8988 | 0.9655 |
| Transformer | 0.8254 | **0.9862** | 0.8987 | 0.9647 |

### Per-Category Detection Rate (Recall with Alarm Persistence)

| Category | CNN | LSTM | MLP | Transformer |
|----------|-----|------|-----|-------------|
| change_weight | 28.8% | 60.8% | 54.1% | **63.1%** |
| drop_object | 59.9% | 83.7% | 82.4% | **87.1%** |
| force_stop | 34.5% | 43.2% | 39.7% | **47.6%** |
| outside_box | **68.3%** | 66.9% | 62.7% | 65.3% |
| push | 57.5% | 58.5% | 71.9% | **79.8%** |
| scale_speed | 32.8% | 49.2% | **50.2%** | 48.4% |
| tcp_wrench | 96.7% | **98.2%** | 97.6% | 98.0% |

---

## 🛠️ How to Run

```bash
# 1. Preprocess raw data and generate X_normal.npy
python preprocess.py

# 2. Train all models (CNN must go first — it saves X_test_normal.npy)
python train_CNN.py
python train_LSTM.py
python train_MLP.py
python train_Transformer.py

# 3. Full evaluation across all models
python evaluate_metrics.py

# 4. Generate label mismatch visualisations
python plot_mismatch.py
```

---

## 📦 Dependencies

- Python 3.8+
- PyTorch
- NumPy, Pandas, scikit-learn
- Matplotlib
- joblib
- RTDE Python Client Library (for live data collection only)