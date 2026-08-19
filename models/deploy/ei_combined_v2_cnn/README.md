# Asset Edge Impulse — CNN (combined_v2)

Generati da `utils/prepare_edge_impulse.py` dal modello ricalibrato in lab.

| | |
|---|---|
| ONNX da importare | `models/ICE/combined_v2\cnn_autoencoder.onnx` |
| Soglia q99 (held-out) | **0.07986** |
| Input del modello | `(1, 50, 24)` float32 |
| Output | scalare = anomaly score (MSE di ricostruzione) |
| Frequenza | 125 Hz (passo 8 ms) |

## Importante

Lo **scaling resta sul PC**: `bridge.py --mode esp32` invia feature già scalate.
L'ESP32 non deve conoscere lo scaler, esegue solo il modello.

Le finestre normali in `test_data/` provengono **solo dalle run di validation**
(held-out dal training), quindi la verifica è onesta.

## Procedura

1. Edge Impulse Studio → nuovo progetto → **Bring Your Own Model (BYOM)**.
   Carica `cnn_autoencoder.onnx`. Il grafo contiene già feature engineering + MSE, quindi
   va importato **intero**: l'output è un singolo scalare (regressione a 1 output),
   non un blocco "Anomaly Detection" nativo di EI.
2. Target device: **Espressif ESP32** (non ESP32-S3: la board è una WROOM-32D).
3. Quantizzazione int8: usa `representative.npy` come dataset di calibrazione.
   Confronta gli score int8 con `expected_scores.csv` — se divergono molto,
   resta in float32.
4. Deployment → **Arduino library** → scarica lo .zip.
5. Estrai in `Documents/Arduino/libraries/`, aggiorna la riga `#include` in
   `models/deploy/firmware/firmware.ino`.
6. Arduino IDE → Board: **ESP32 Dev Module** → Upload.
7. Verifica di accettazione:
   `python utils/verify_mcu.py --serial-port COMx --assets models\deploy\ei_combined_v2_cnn`
8. Imposta `anomaly_threshold = 0.07986f` nel firmware.

## File

- `test_data/normal.NNN.csv`, `test_data/anomaly.NNN.csv` — finestre in formato CSV EI
- `expected_scores.csv` — score di riferimento (onnxruntime): la verità di confronto
- `representative.npy` — 500 finestre normali per la calibrazione int8, shape (500, 50, 24)
- `representative_flat.npy` — le stesse, appiattite a (500, 1200): prova questa se EI rifiuta il tensore 3D
