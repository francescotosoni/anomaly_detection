# Collaborative Robot (UR5) Time-Series Anomaly Detection

This repository contains an advanced **Time-Series Anomaly Detection (TSAD)** pipeline for Universal Robots (UR5) collaborative manipulators. It is designed to evaluate both deep learning architectures (1D CNNs and LSTMs) using unsupervised learning techniques, aligning with the evaluation methodologies of academic literature (such as Graabæk et al.'s *stochastic_PnP* dataset) while maintaining robust metrics suitable for industrial deployment.

---

## 📂 Repository Structure

The core pipeline is composed of the following files:

- **`preprocess.py`**
  Handles data ingestion, extracts the relevant kinematic/current features from raw `.csv` files, standardizes the data using a `RobustScaler` (trained solely on healthy trajectories), and exports scaled windowed tensors.
- **`train_LSTM.py`**
  Trains an LSTM (Long Short-Term Memory) Autoencoder. Given the inertial nature of robotic movement over time, this architecture significantly outperforms others at identifying long-term sequences while being highly robust against natural vibration-induced false positives.
- **`train_CNN.py`**
  Trains a 1D Convolutional Neural Network (CNN) Autoencoder featuring **Dilated Convolutions**. Highly reactive to sudden shocks but inherently lacks the "long-term memory" required for stable whole-file validation. Used primarily for immediate geometric impact evaluation.
- **`evaluate_metrics.py`**
  The backbone of the project. A comprehensive validation script validating the models effectively over industrial scenarios (File-Level) and academic benchmarks (Window-Level) using state-of-the-art TSAD interpretations.


---

## 🧠 Model Training & The Unsupervised Approach

Both the CNN and LSTM act as **Autoencoders**. During training, the models are fed purely *Healthy (Normal)* kinematic data and are tasked with compressing and subsequently reconstructing these trajectories. 
During inference, if the collaborative robot is pushed, hits an obstacle, or drops a payload, the autoencoder fails to reconstruct these unfamiliar movements, causing the **Reconstruction Error** to spike.

---

## ⚙️ Advanced Implementation Details

To solve the significant gap between the physics of the UR5 and statistical performance metrics, several key implementation adjustments were applied:

### 1. Feature-Max Loss Aggregation (Advanced MSE)
Typically, calculating the Mean Squared Error (MSE) across the time axis *and* feature axis dilutes isolated anomaly spikes. Instead, the loss computation is defined as:
```python
errors = loss.mean(dim=Time).max(dim=Features)
```
This averages the error dynamically across the time axis (for stability) but takes the **maximum error across all 18 features**. If a collision sharply alters the current of a single joint motor, the alarm triggers accurately without being masked by the normal state of the other five motors.

### 2. File-Max Statistical Thresholds (File-Level)
In most architectures, establishing an anomaly threshold based on the *frame* error distribution generates massive **False Positives** when deployed industrially over longer traces. Every long sequence inevitably contains physiological vibration spikes.
To counter this, the evaluation threshold isn't the 99th percentile of frames, but the **95th percentile of the maximum error per whole file** calculated strictly on the `X_normal` array. This saves the *Precision* of the autoencoder.

### 3. High-Fidelity Sync (`STEP_SIZE = 1`)
Using overlapping windows is standard, but utilizing large shifts (e.g., `STEP = 10`) creates catastrophic prediction delays relative to hardware tags (e.g., up to ~250ms blind-spots). 
By enforcing `STEP_SIZE = 1`, inference tracks the UR5 hardware logs millisecond-by-millisecond, perfectly mapping to the onset of the mechanical shock. 

### 4. Point-Adjusted Evaluation (Window-Level)
Physical collisions last fractions of a second, resulting in a momentary reconstruction error spike. However, the hardware flag (`actual_digital_input_bits == 32`) often persists long after the robot has reached a steady/safe state. Traditional frame-by-frame metric evaluations heavily penalize the model in these instances when the error drops back to normal.
To combat this, the pipeline incorporates the **Point-Adjusted (PA)** protocol:
> *If an alarm correctly triggers at least once during a continuous hardware anomaly segment, the entire segment is considered successfully detected. False negatives within the post-shock steady state are retroactively nullified, keeping the ROC-AUC and F1-Scores scientifically representative (e.g., scoring ~0.96 ROC-AUC).*

---

## 📊 Summary of Validated Results
Under the combined metrics, the **LSTM Autoencoder** performs optimally across both spectra:
* **Industrial Validation (File-Level)**: F1-Score ~ 0.91 (Precision 0.94 / Recall 0.88). Excellent reliability with minimized false positives.
* **Academic Benchmark (Window-Level PA)**: ROC-AUC ~ 0.96. Matching or exceeding benchmark results found in the reference UR5 *stochastic_PnP* documentation.