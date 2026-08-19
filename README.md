# UR5 Collaborative Robot — Time-Series Anomaly Detection

Unsupervised **Time-Series Anomaly Detection (TSAD)** for Universal Robots UR5 collaborative
manipulators, from architecture comparison on a public benchmark through to a working
deployment on a microcontroller supervising a live robot.

The project has two phases, both contained in this repository:

| Phase | Data | Goal | Outcome |
|---|---|---|---|
| **1. Benchmark** | Graabæk et al. public dataset | Compare four autoencoder architectures | MLP best overall, ROC-AUC 0.826 |
| **2. Deployment** | Own recordings on a physical UR5 | Run detection on an MCU, in real time | MLP on ESP32-S3, 13.7 ms, ROC-AUC 0.957 / 0.981 |

---

## Repository structure

```
ur5_data_collection/
├── anomaly_config_24.xml        RTDE output recipe (24 features) — required by bridge and collect
├── lab_retrain.py               Training and threshold calibration on lab recordings
│
├── models/
│   ├── train_CNN.py             1D CNN autoencoder (+ CNNUpAutoencoder variant)
│   ├── train_LSTM.py            LSTM autoencoder
│   ├── train_MLP.py             MLP autoencoder on tabular statistics
│   ├── train_Transformer.py     Lightweight Transformer autoencoder
│   │
│   ├── ICE/combined_v2/         Deployed model: ONNX, thresholds, scaler, manifest, metrics
│   │
│   └── deploy/
│       ├── bridge.py            RTDE → inference → dashboard, PC or MCU mode
│       ├── firmware/            ESP32-S3 / Arduino Nano 33 BLE firmware
│       ├── dashboard/           Live web dashboard (WebSocket)
│       └── ei_combined_v2_mlp/  Edge Impulse assets and deployment verification
│
└── utils/
    ├── lab_io.py                Canonical feature order, CSV reading, path resolution
    ├── collect.py               Live acquisition via RTDE
    ├── pick_and_place.py        Task execution and waypoint teaching (--task 1|2)
    ├── label_lab_anomalies.py   Anomaly labelling from TCP force
    ├── evaluate_lab.py          Event and window metrics, threshold sweeps
    ├── prepare_edge_impulse.py  Test vectors and int8 calibration set
    ├── patch_ei_gather.py       Unblocks the Gather operator in the Edge Impulse SDK
    ├── export_to_onnx.py        ONNX wrappers (score computed inside the graph)
    ├── preprocess.py            Ingestion, scaling, windowing
    ├── feature_engineering.py   Statistical feature extraction for the MLP
    └── evaluate_metrics.py      Benchmark-phase evaluation
```

Datasets, the RTDE client library, generated figures and a number of exploratory scripts are
deliberately untracked; see [Reproducing the results](#reproducing-the-results).

---

## Method

All models are **autoencoders trained exclusively on healthy data**. The network learns to
reconstruct normal motion; anomalous input reconstructs poorly, so the **reconstruction error**
rises above a calibrated threshold. No anomaly labels are needed for training, which is what
makes the approach transferable to a new cell without re-labelling.

**Features.** 24 physical channels streamed from the UR5 controller at 125 Hz:
joint positions `actual_q_[0-5]`, joint currents `actual_current_[0-5]`, joint velocities
`actual_qd_[0-5]`, TCP forces `actual_TCP_force_[0-5]`.

**Windowing.** 50 samples (400 ms), step 1 sample — 98% overlap, millisecond-resolution scoring.

**Scaling.** A category-aware scaler normalises each physical group (positions, currents,
velocities, forces) with its own statistics, so that units of different magnitude contribute
comparably. Scaler parameters are fitted on training runs only and shipped alongside the model
in `scaler.json`.

**Sequential models** (CNN, LSTM, Transformer) consume the raw `(batch, 50, 24)` tensor.
**The MLP** instead drops five uninformative channels (`actual_current_3`, `actual_qd_[2-5]`,
identified by feature-importance analysis) and collapses each window into five statistical
moments per channel — mean, standard deviation, max, min, RMS — giving a dense `19 × 5 = 95`
vector. This changes the paradigm from sequence reconstruction to *statistical footprint*
reconstruction.

---

## Architectures

### 1D CNN autoencoder — 46,968 parameters

```
Encoder:  Conv1d(24→64, k=3)                → BN → ReLU → Dropout(0.1)
          Conv1d(64→64, k=3, dilation=2)    → BN → ReLU → MaxPool(2)
          Conv1d(64→32, k=3, dilation=2)    → BN → ReLU → MaxPool(5)

Decoder:  ConvTranspose1d(32→64, k=5, s=5)  → BN → ReLU      # 5 → 25
          ConvTranspose1d(64→64, k=2, s=2)  → BN → ReLU      # 25 → 50
          Conv1d(64→24, k=3)
```

Dilated convolutions capture multi-scale temporal structure at constant parameter cost.

`train_CNN.py` also provides **`CNNUpAutoencoder`**, identical except that the decoder replaces
both transposed convolutions with nearest-neighbour upsampling followed by convolution. The two
have exactly the same parameter count. The variant exists because ESP-NN, Espressif's accelerated
kernel library, implements no transposed-convolution kernel — see
[Microcontroller deployment](#microcontroller-deployment). It is written with explicit
height-1 2D operators, because the ONNX-to-TFLite converter mishandles 1D `Resize` and inflates
the intermediate tensors by a factor of 64.

### LSTM autoencoder — 34,616 parameters

```
Encoder: LSTM(24→32, layers=2, dropout=0.2) → h_n[-1]        # true 1200→32 bottleneck
Bridge:  Linear(32→32) + ReLU
Repeat:  unsqueeze(1).repeat(seq_len)
Decoder: LSTM(32→32, layers=2, dropout=0.2) → Linear(32→24)
```

### MLP autoencoder — 45,311 parameters

```
(batch, 50, 24) → drop 5 channels → tabular statistics → (batch, 95)
Encoder: 95 → 128 → 64 → [32]
Decoder: [32] → 64 → 128 → 95
```

### Transformer autoencoder — 144,440 parameters

```
Linear(24→64) + learnable positional encoding (50×64)
TransformerEncoder [2 layers, 2 heads, ff=128] → mean-pool → Linear(64→32)   # bottleneck
Linear(32→64) → repeat → TransformerDecoder [2 layers, 2 heads, ff=128] → Linear(64→24)
```

---

## Phase 1 — Public benchmark

Evaluated against the dataset and protocol of Graabæk et al. Per-timestep ROC-AUC, obtained by
averaging the scores of all windows overlapping each sample. The `force_stop` category is
excluded, as it was not evaluated by the original authors.

| Category | CNN | LSTM | MLP | Transformer |
|---|---|---|---|---|
| change_weight | 0.5804 | 0.6285 | **0.6520** | 0.5353 |
| drop_object | 0.8569 | **0.9313** | 0.8981 | 0.8089 |
| outside_box | 0.8464 | 0.8582 | **0.8606** | 0.7411 |
| push | 0.6439 | 0.6337 | **0.6751** | 0.6592 |
| scale_speed | 0.6126 | **0.6752** | 0.6665 | 0.5957 |
| tcp_wrench | **0.9318** | 0.8625 | 0.8938 | 0.8740 |
| **All anomalies** | 0.7719 | 0.8075 | **0.8260** | 0.7406 |

Run with `python utils/evaluate_metrics.py`.

---

## Phase 2 — Deployment on a physical UR5

### Data collection

Two distinct pick-and-place tasks were recorded on a real UR5, differing in pick and place
station geometry. Two tasks rather than one, so that cross-task evaluation can show whether the
model generalises beyond a single memorised trajectory.

`utils/pick_and_place.py --task 1|2` selects the waypoint set for both teaching and execution;
`utils/collect.py` streams and records the 24 channels via RTDE.

| | Normal runs | Anomaly recordings | Annotated events |
|---|---|---|---|
| Task 1 | 26 | 9 | 31 |
| Task 2 | 15 | 4 | 14 |

Anomalies were induced physically: manual pushes against the arm during motion (`push_*`) and
extra payload attached to the end effector (`weight_*`).

### Labelling without ground truth

The public dataset marks anomaly injection on a digital input. The lab recordings have no such
signal, so `utils/label_lab_anomalies.py` derives labels from the TCP force magnitude:
`weight_*` recordings are labelled as distributed anomalies over the whole file, `push_*` as
localised events around force peaks.

Two label variants are produced. The **primary** variant, with event merging, is used for
event-based recall; the **tight** variant, with minimal dilation, is used for window-level
ROC-AUC, where persistent labels would otherwise inflate the score. Events truncated by the
start or end of a recording are excluded from recall.


### Training and calibration

`lab_retrain.py` reuses the architectures above and adds what deployment requires:

- **Per-run splitting.** With 98% window overlap, a random window split would leak nearly
  identical windows between training and validation. Runs are split instead, and the scaler is
  fitted only on training runs.
- **Anomalous files excluded by construction** — a normal-run glob plus a rejection list, so an
  anomaly recording cannot silently enter the training set and teach the autoencoder to
  reconstruct it.
- **Threshold computed through the exported ONNX wrapper**, not from the training loop. The two
  aggregate the reconstruction error differently, and the discrepancy is large enough to make a
  threshold taken from training unusable at deployment.
- **A reproducible manifest** recording the exact train/validation split, scaler parameters,
  validation loss and convergence.

The deployed model, `models/ICE/combined_v2/`, was trained on both tasks: 41 normal runs,
31 for training and 10 held out (241,401 and 79,904 windows respectively).

### Operating point

The threshold is the **q-th quantile of the reconstruction error over held-out normal runs**.
It never uses anomalous data, so the quantile is a design parameter — the tolerated
false-alarm rate — rather than a value fitted on test data.

| Quantile | Threshold | Expected false alarms per 1000 windows |
|---|---|---|
| q99 | 0.01385 | 10.0 |
| q99.5 | 0.01715 | 5.0 |
| **q99.95** | **0.02733** | **0.5** |

q99 proved too tight in practice. False alarms concentrated 95–99% in the slowest quartile of
motion — object release and the pause at the home position — where the score distribution shifts
upwards for both classes. Measured per regime, the model separates normal from anomalous just as
well when the robot is slow, so this is a threshold-placement effect, not a modelling failure.
q99.95 was adopted.

### Results

Model `combined_v2`, MLP, alarm protocol with 40 ms debouncing.

| | Task 1 | Task 2 |
|---|---|---|
| **ROC-AUC** | **0.9573** | **0.9807** |
| **PR-AUC** | 0.9261 | 0.9612 |

Event-based, at the three operating points:

| Quantile | Task 1 precision | recall | F1 | Task 2 precision | recall | F1 |
|---|---|---|---|---|---|---|
| q99 | 0.574 | 1.000 | 0.729 | 0.467 | 1.000 | 0.636 |
| q99.5 | 0.705 | 1.000 | 0.827 | 0.560 | 1.000 | 0.718 |
| **q99.95** | **0.967** | 0.935 | **0.951** | **1.000** | 0.929 | **0.963** |

Moving from q99 to q99.95 reduces false alarms from 23 to 1 on task 1 and from 16 to 0 on
task 2, at the cost of three events out of 45.

ROC-AUC and PR-AUC are the figures to compare across models and setups, being independent of
both threshold and alarm protocol. Precision, recall and F1 should always be quoted together
with the quantile and protocol that produced them.

---

## Microcontroller deployment

The model is exported to ONNX with the anomaly score computed **inside the graph**, so the board
receives scaled features and returns a scalar. Quantisation to int8 and code generation are done
with Edge Impulse; the impulse must be declared as **regression**, since the anomaly-detection
path is not int8-aware and silently returns zero.

`utils/prepare_edge_impulse.py` produces the test vectors, the expected scores and the int8
calibration set. `utils/patch_ei_gather.py` re-enables the `Gather` operator, which the SDK
compiles out on ESP32 targets while the generated resolver still references it.

### Measured on ESP32-S3

| | MLP | CNN |
|---|---|---|
| Inference | **13.7 ms** | 644 ms |
| Tensor arena | 11.7 KB | 31.5 KB |
| Decision agreement with PC | **100/100** | 100/100 |
| Score correlation with PC | 0.9966 | 0.9998 |

At 125 Hz with inference every 10 samples the MLP occupies 17% of the available time.

The CNN is unusable on this target. The cause is not the SDK overhead: a plain scalar C loop
performing the same 1.9 M multiply-accumulates on the same board takes 103 ms, and the model has
more operators than the MLP yet the MLP runs in 13.7 ms. ESP-NN provides accelerated kernels for
convolution, depthwise convolution, fully-connected, pooling and elementwise operations, **but
none for transposed convolution**, so the decoder falls back to the TensorFlow Lite Micro
reference implementation. `CNNUpAutoencoder` addresses this at the architectural level; measured
on the same evaluation it loses about 1% of ROC-AUC, but the MLP remains both faster and more
accurate, so it is the deployed model.

### Threshold under quantisation

Input quantisation injects noise, and because the score *is* the reconstruction error, that
noise inflates it. The float32 threshold is therefore not valid on the board and is recalibrated
on the quantised model: **0.03896** for the MLP at q99.95.

A second effect matters. Edge Impulse derives the int8 output range from the calibration set; a
set containing only normal windows produces a range barely above the threshold, so every anomaly
saturates. That not only makes severity unreadable, it makes any quantile above q99.5 land
exactly on full scale and become unusable. `prepare_edge_impulse.py --score-cap` extends the
calibration set with mild anomalies so the output range reaches a chosen multiple of the
threshold.

### Live system

`models/deploy/bridge.py` connects to the robot over RTDE, scales the features and either runs
inference on the PC (`--mode pc`) or forwards them to the board (`--mode esp32`), streaming
scores to a web dashboard over WebSocket. It reconnects automatically after a protective stop.

The dashboard shows the live score, a colour-filled threshold banner and an event counter, with
configurable time-based debouncing. On boards with an addressable LED the firmware mirrors it:
blue while waiting for data, green below threshold, red above.

Note that the two modes require **different thresholds** — 0.0273 for float32 on the PC, 0.0390
for int8 on the board — because the score distributions differ.

---

## Reproducing the results

```bash
# Phase 1 — public benchmark
python utils/preprocess.py
python models/train_CNN.py          # and train_LSTM.py, train_MLP.py, train_Transformer.py
python utils/evaluate_metrics.py

# Phase 2 — lab recordings
python utils/collect.py                                     # acquisition (robot required)
python utils/label_lab_anomalies.py                         # labels from TCP force
python lab_retrain.py --data data/lab/task1 data/lab/task2 \
    --model mlp --val-runs 10 --out models/ICE/combined_v2
python utils/evaluate_lab.py --model-dir models/ICE/combined_v2 \
    --eval-task 1 --models mlp --debounce-ms 40

# Deployment
python utils/prepare_edge_impulse.py --model-dir models/ICE/combined_v2 \
    --model mlp --score-cap 4
# ... export from Edge Impulse as regression / int8 / Arduino library ...
python utils/patch_ei_gather.py --all
python models/deploy/bridge.py --robot-ip <IP> --mode esp32 \
    --serial-port <PORT> --scaler models/ICE/combined_v2/scaler.json
```

**Not tracked in this repository.** Recordings (`data/`), the RTDE client library, generated
figures and reports, the experimental per-task models, and a set of exploratory and
deploy-verification scripts. The lab recordings in particular are not reproducible without the
physical cell, and are archived separately.

---

## References

- Graabæk et al., *An Experimental Comparison of Anomaly Detection Methods for Collaborative
  Robot Manipulators* — [IEEE](https://ieeexplore.ieee.org/abstract/document/10160007)
- Dataset — [Physical Anomaly Detection Dataset for UR5](https://zenodo.org/records/5849300)
