# Collaborative Robot (UR5) Time-Series Anomaly Detection

This repository contains an advanced **Time-Series Anomaly Detection (TSAD)** pipeline for Universal Robots UR5 collaborative manipulators. The project evaluates and compares four deep learning architectures — 1D CNN, LSTM, MLP, and Transformer — using unsupervised autoencoder-based learning, targeting both academic benchmarks (Graabæk et al.) and **physical MCU deployment via TinyML**.

---

## 📂 Repository Structure

```
ur5_data_collection/
│
├── anomaly_config.xml       # Robot anomaly configuration
├── README.md                # This documentation
│
├── data/                    # Dataset (stochastic_PnP trajectories)
├── RTDE_Python_Client_Library/ # UR5 RTDE Interface
│
├── models/                  # Neural Network Architectures & Training
│   ├── train_CNN.py         # 1D CNN Autoencoder (Dilated Conv + Dropout)
│   ├── train_LSTM.py        # LSTM Autoencoder (true bottleneck via RepeatVector)
│   ├── train_MLP.py         # MLP Autoencoder (tabular statistical features)
│   └── train_Transformer.py # Lightweight Transformer Autoencoder
│
└── utils/                   # Data Processing & Evaluation
    ├── collect.py           # Live data collection via RTDE
    ├── preprocess.py        # Data ingestion, RobustScaler, windowing
    ├── feature_engineering.py # Statistical extraction for MLP
    ├── check_data.py        # Dataset validation and integrity checks
    ├── evaluate_metrics.py  # Full evaluation (Deploy vs Academic vs Per-Category)
    └── plot_mismatch.py     # Visualization of the label mismatch problem
```

---

## 🧠 Approach: Unsupervised Autoencoder-Based Detection

All four models are trained as **Autoencoders** on purely *healthy (normal)* UR5 kinematic data. During training the model learns to compress and reconstruct normal movement patterns. At inference time, anomalous inputs (impacts, unexpected forces, weight changes, etc.) cannot be reconstructed accurately, causing the **Reconstruction Error** to spike above a calibrated threshold.

This unsupervised approach requires **no anomaly labels during training**, making it deployable on new robot setups without re-labeling.

**Input Features & Feature Engineering:**
The dataset natively contains 24 physical features (recorded directly from the UR5 controller):
- Joint positions: `actual_q_[0-5]`
- Joint currents: `actual_current_[0-5]`
- Joint velocities: `actual_qd_[0-5]`
- TCP forces: `actual_TCP_force_[0-5]`

**Sequential Models (CNN, LSTM, Transformer):** 
These models ingest **all 24 features** in their raw temporal format `(batch, 50, 24)`. This preserves the full physical context of the robot's kinematics and aligns perfectly with the academic methodology of Graabæk et al., allowing the models to learn complex cross-sensor temporal correlations (like the relation between velocity and force during an impact).

**Tabular Model (MLP):** 
Standard MLPs struggle with raw high-dimensional sequences. To solve this, the MLP utilizes a custom feature engineering pipeline:
1. It explicitly drops 5 noisy/uninformative features (`actual_current_3` and `actual_qd_[2-5]`), as identified by AutoGluon feature importance analysis, reducing the base features to 19.
2. It collapses the 50-timestep temporal window into 5 statistical moments (Mean, Std, Max, Min, RMS) for each of the 19 features.
3. This creates a highly dense, static vector of `19 × 5 = 95` tabular features per window.

**Windowing:** `WINDOW_SIZE = 50` samples, `STEP_SIZE = 1` (overlapping, millisecond-level resolution, fully aligned with academic baselines).

---

## 🏗️ Model Architectures

### 1D CNN Autoencoder (`train_CNN.py`) — 46,968 params / 183 KB

```
Encoder:
  Conv1d(24→64, k=3)     → ReLU → Dropout(0.1)
  Conv1d(64→64, k=3, dilation=2) → ReLU → MaxPool(2)
  Conv1d(64→32, k=3, dilation=2) → ReLU → MaxPool(5)

Decoder:
  Upsample(×5) → Conv1d(32→64, dilation=2) → ReLU
  Upsample(×2) → Conv1d(64→64) → ReLU
  Conv1d(64→24)
```

Dilated convolutions capture multi-scale temporal patterns without increasing parameter count. Dropout prevents overfitting on the normal distribution. 

---

### LSTM Autoencoder (`train_LSTM.py`) — 34,616 params / 135 KB

```
Encoder: LSTM(24→32, layers=2, dropout=0.2)
         → only h_n[-1]: (batch, 32)   ← true bottleneck (1200→32 values)
Bridge:  Linear(32→32) + ReLU
Repeat:  unsqueeze(1).repeat(seq_len)  ← RepeatVector pattern
Decoder: LSTM(32→32, layers=2, dropout=0.2)
Output:  Linear(32→24)
```

Extracts only the last hidden state `h_n[-1]`, achieving a genuine 1200→32 bottleneck. Outstanding performance on point-impact anomalies (`drop_object`, `tcp_wrench`).

---

### MLP Autoencoder (`train_MLP.py`) — 45,311 params / 177 KB

```
Input: (batch, 50, 24) → Feature Drop (19) → Tabular Stats (Mean, Std, Max, Min, RMS) → (batch, 95)
Encoder: 95 → 128 → 64 → [32]   (bottleneck)
Decoder: [32] → 64 → 128 → 95
Output: reshape to match the 95 tabular features
```

**Implementation Differences:** Unlike the sequential models, the MLP does not reconstruct a temporal sequence. By collapsing the temporal dimension into statistical moments, it completely changes the anomaly detection paradigm from "sequence reconstruction" to "statistical footprint reconstruction". This drastically reduces the parameter count and smooths out the baseline errors, allowing it to achieve the highest aggregate ROC-AUC, albeit shifting away from the pure academic comparison of Graabæk et al.

---

### Transformer Autoencoder (`train_Transformer.py`) — 144,440 params / 564 KB

```
Input (24) → Linear(64) + Learnable PosEncoding(50×64)
           → TransformerEncoder [2 layers, 2 heads, ff=128, dropout=0.1]
           → MeanPool over time → (batch, 64)
           → Linear(64→32) + ReLU          ← bottleneck
           → Linear(32→64) + ReLU + Repeat ← expand
           → TransformerDecoder [2 layers, 2 heads, ff=128, dropout=0.1]
           → Linear(64→24)
```

Self-attention allows the model to capture global temporal dependencies. It is the heaviest model but still fits within the constraints of mid-tier MCUs.

---

## ⚙️ Training Pipeline (Shared Across All Models)

All four models use a mathematically rigorous training pipeline aligned with academic literature:

| Component | Configuration |
|-----------|---------------|
| **Loss function** | `nn.MSELoss()` (Standard Mean Squared Error) |
| **Optimiser** | Adam, lr=0.001 |
| **LR Scheduler** | `CosineAnnealingWarmRestarts` (T_0=15, T_mult=2) |
| **Gradient clipping** | `max_norm=1.0` |
| **Early stopping** | Patience = 15 epochs |
| **Max epochs** | 150 |
| **Batch size** | 64 |
| **Data split** | 70% Train / 10% Val / 20% Test |
| **Threshold** | q99 percentile of validation reconstruction errors |

---

## 📊 Evaluation Framework (`evaluate_metrics.py`)

Three evaluation paradigms are reported for each model:

### [1] Academic Benchmark: Per-Timestep Scoring (Table C)
Matches the methodology of Graabæk et al. The anomaly score of a window is the MSE of the reconstructed window (scalar). Then, for each timestep, the anomaly scores of the 50 overlapping windows are averaged. This provides a continuous, millisecond-level anomaly signal.

### [2] MCU Deploy Metrics — Alarm Persistence *(most realistic)*
Simulates actual MCU deployment behaviour: once a window exceeds the threshold, the alarm remains active for **200 subsequent windows (= 2.0 seconds at 100 Hz)**. This reflects how an embedded system handles transient physical impacts.

### [3] Point-Adjusted (PA)
Standard in TSAD literature: if at least one window in a continuous anomalous segment is correctly flagged, the entire segment counts as detected. Represents the theoretical upper-bound.

---

## 📈 Results Summary

### Per-Category ROC-AUC (Academic Benchmark — Per-Timestep)
*Directly comparable to Graabæk et al. Note: `force_stop` is a custom anomaly not present in the original paper. Because Autoencoders perfectly reconstruct a static, stopped robot (MSE ~0), they naturally fail to detect a `force_stop` using pure MSE, which explains the ~0.50 score.*

| Category | CNN | LSTM | MLP | Transformer |
|----------|-----|------|-----|-------------|
| change_weight | 0.5804 | 0.6285 | 0.6520 | 0.5353 |
| drop_object | 0.8569 | **0.9313** | 0.8981 | 0.8089 |
| force_stop | 0.5070 | **0.5657** | 0.5498 | 0.6085 |
| outside_box | 0.8464 | 0.8582 | **0.8606** | 0.7411 |
| push | 0.6439 | 0.6337 | **0.6751** | 0.6592 |
| scale_speed | 0.6126 | **0.6752** | 0.6665 | 0.5957 |
| tcp_wrench | **0.9318** | 0.8625 | 0.8938 | 0.8740 |

### MCU Deploy Metrics — Alarm Persistence 2.0s

| Model | Size (KB) | Precision | Recall | **F1** |
|-------|-----------|-----------|--------|--------|
| CNN | 183 | 0.2972 | 0.6346 | 0.4048 |
| LSTM | **135** | 0.3158 | 0.7410 | 0.4428 |
| **MLP** | 177 | 0.3275 | **0.8107** | **0.4665** 🥇 |
| Transformer | 564 | **0.3287** | 0.7909 | 0.4644 |

### Point-Adjusted (PA) Benchmark

| Model | Precision | Recall | **F1** | ROC-AUC |
|-------|-----------|--------|--------|---------|
| CNN | **0.6054** | 0.9845 | 0.7498 | 0.9644 |
| LSTM | 0.6028 | 0.9948 | 0.7507 | 0.9713 |
| MLP | 0.5984 | **1.0000** | 0.7487 | **0.9867** 🥇 |
| Transformer | 0.6048 | 0.9970 | **0.7529** 🥇 | 0.9692 |

---

## 🛠️ How to Run

```bash
# 1. Preprocess raw data and generate X_normal.npy
python utils/preprocess.py

# 2. Train all models
python models/train_CNN.py
python models/train_LSTM.py
python models/train_MLP.py
python models/train_Transformer.py

# 3. Full evaluation across all models
python utils/evaluate_metrics.py
```

---

## 📚 References & Acknowledgements

This project was built to validate and expand upon the academic baseline established by Graabæk et al.:

- **Original Paper**: [An Experimental Comparison of Anomaly Detection Methods for Collaborative Robot Manipulators](https://ieeexplore.ieee.org/abstract/document/10160007)
- **Dataset**: [Physical Anomaly Detection Dataset for UR5](https://zenodo.org/records/5849300)