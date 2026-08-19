"""
UR5 Anomaly Detection — Lab Bridge
===================================
Bridge between the UR5 robot, the ESP32 (optional) and the web dashboard.

Two operating modes:
  --mode pc     : ONNX inference directly on the PC (no ESP32 needed)
  --mode esp32  : sends data to the ESP32 over serial and reads the score back

Usage:
  python bridge.py --robot-ip 192.168.1.100 --mode pc
  python bridge.py --robot-ip 192.168.1.100 --mode esp32 --serial-port COM3
"""

import sys
import os
import time
import json
import struct
import argparse
import threading
import logging
from collections import deque
import sys
import os

# Add the utils folder to the path so collect.py can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))
try:
    from collect import StateRecorder
except ImportError:
    StateRecorder = None

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'RTDE_Python_Client_Library'))
sys.path.insert(0, PROJECT_ROOT)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('bridge')

# --- Constants ---
RTDE_PORT = 30004
FREQUENCY = 125               # Hz (CB3 = 125, e-Series = 500)
WINDOW_SIZE = 50               # Timesteps per finestra
N_FEATURES = 24                # Features per timestep
INFERENCE_STEP = 10            # Ogni quanti campioni fare inferenza
DASHBOARD_FPS = 30             # Max update rate per la dashboard
WEBSOCKET_PORT = 8765          # Porta WebSocket per la dashboard
SERIAL_BAUD = 460800           # Baud rate per ESP32

# Feature order (deve corrispondere a anomaly_config_24.xml)
FEATURE_NAMES = [
    'actual_q',           # 0-5:   Joint positions (rad)
    'actual_current',     # 6-11:  Joint currents (A)
    'actual_qd',          # 12-17: Joint velocities (rad/s)
    'actual_TCP_force',   # 18-23: TCP forces/torques (N/Nm)
]

# ---- Scaler ----
# The model was trained on data normalised with these parameters.
# Formula: X_scaled = (X_raw - mean) / std
# WITHOUT this normalisation the model gives completely wrong results.
#
# The parameters live in an EXTERNAL JSON file (scaler.json) next to the ONNX,
# model and scaler stay coupled as data and this source file is never patched by
# retraining. The values below are only a FALLBACK used when the JSON is missing
# (e.g. the original training model).
DEFAULT_SCALER_PARAMS = {
    'q':       {'mean': -0.960207, 'std': 1.223060, 'slice': slice(0, 6)},
    'current': {'mean':  0.925601, 'std': 1.419957, 'slice': slice(6, 12)},
    'qd':      {'mean': -0.000006, 'std': 0.226494, 'slice': slice(12, 18)},
    'tcp':     {'mean': -0.153784, 'std': 6.668897, 'slice': slice(18, 24)},
}


def resolve_scaler_path(args):
    """
    Determine which scaler.json to use.

    The scaler is independent of WHERE inference runs: the bridge always needs it,
    because it scales the features before sending them (to the board or to the
    local ONNX model). Hence a dedicated --scaler argument: in esp32 mode the ONNX
    is not used for inference, and deriving the scaler from it would be an
    arbitrary coupling.
    Precedence: explicit --scaler > folder of --onnx-model.
    If it cannot be determined we EXIT with an error rather than falling back to
    defaults: a wrong scaler makes nothing fail, it simply produces out-of-scale
    (saturated) scores that look like real anomalies -- the worst kind of fault,
    because it is silent and believable.
    """
    if args.scaler:
        return os.path.abspath(args.scaler)
    if args.onnx_model:
        return os.path.join(os.path.dirname(os.path.abspath(args.onnx_model)), 'scaler.json')
    log.error("Impossibile determinare lo scaler: specifica --scaler (o --onnx-model).")
    log.error("  Esempio: --scaler models/ICE/combined/scaler.json")
    sys.exit(1)


def load_scaler_params(scaler_path):
    """
    Load the scaling parameters from scaler.json. The JSON stores the slice as a
    [start, stop] list, converted back to a `slice` here.
    """
    if not os.path.exists(scaler_path):
        log.error(f"Scaler not found: {scaler_path}")
        log.error("  The scaler MUST match the one used to train the model:")
        log.error("  with wrong parameters the scores saturate and look like a continuous anomaly.")
        sys.exit(1)
    try:
        with open(scaler_path, 'r') as f:
            raw = json.load(f)
        params = {}
        for cat, p in raw.items():
            start, stop = p['slice']
            params[cat] = {'mean': p['mean'], 'std': p['std'], 'slice': slice(start, stop)}
        log.info(f"Scaler loaded from: {scaler_path}")
        # The benchmark models in models/onnx/ use the academic dataset scaler:
        # on real robot data it produces saturated scores (domain shift).
        if os.path.basename(os.path.dirname(scaler_path)) == 'onnx':
            log.warning("WARNING: you are using the ACADEMIC dataset scaler "
                        "(models/onnx/). On your robot data the scores will saturate.")
            log.warning("  For the model recalibrated in the lab use: "
                        "--scaler models/ICE/combined/scaler.json")
        return params
    except Exception as e:
        log.error(f"Impossibile leggere {scaler_path}: {e}")
        sys.exit(1)


# Populated at start-up in run_bridge(), based on the selected ONNX model.
SCALER_PARAMS = DEFAULT_SCALER_PARAMS


def scale_features(raw_features):
    """
    Normalise the 24 raw RTDE features using the same parameters as training
    (CustomCategoryScaler saved in data/scaler.pkl).
    
    Input:  list of 24 raw floats from the robot
    Output: list of 24 normalised floats, ready for the model
    """
    scaled = list(raw_features)  # copia
    for cat, params in SCALER_PARAMS.items():
        slc = params['slice']
        mean = params['mean']
        std = params['std']
        for i in range(slc.start, slc.stop):
            scaled[i] = (scaled[i] - mean) / std
    return scaled


class RingBuffer:
    """Ring buffer accumulating windows of data."""
    def __init__(self, window_size, n_features):
        self.window_size = window_size
        self.n_features = n_features
        self.buffer = [[0.0] * n_features for _ in range(window_size)]
        self.write_idx = 0
        self.total_samples = 0
    
    def push(self, features):
        """Append one sample (list of n_features floats)."""
        self.buffer[self.write_idx] = list(features)
        self.write_idx = (self.write_idx + 1) % self.window_size
        self.total_samples += 1
    
    @property
    def ready(self):
        """True se abbiamo accumulato almeno window_size campioni."""
        return self.total_samples >= self.window_size
    
    def get_window_flat(self):
        """Return the window as a flat array (1200 floats), in chronological order."""
        flat = []
        start = self.write_idx  # il più vecchio
        for i in range(self.window_size):
            idx = (start + i) % self.window_size
            flat.extend(self.buffer[idx])
        return flat
    
    def get_window_2d(self):
        """Return the window as a list of lists (50 x 24)."""
        window = []
        start = self.write_idx
        for i in range(self.window_size):
            idx = (start + i) % self.window_size
            window.append(self.buffer[idx])
        return window


class OnnxInferenceEngine:
    """ONNX inference on the PC (--mode pc)."""
    def __init__(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError:
            log.error("onnxruntime not installed. Install with: pip install onnxruntime")
            sys.exit(1)
        
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        log.info(f"ONNX model loaded: {model_path}")
        log.info(f"  Input: {self.session.get_inputs()[0].shape}")
        log.info(f"  Output: {self.session.get_outputs()[0].shape}")
    
    def predict(self, window_flat):
        """Run inference. window_flat = list of 1200 floats."""
        import numpy as np
        data = np.array(window_flat, dtype=np.float32).reshape(1, WINDOW_SIZE, N_FEATURES)
        result = self.session.run(None, {self.input_name: data})
        score = float(result[0][0][0])
        return score


class SerialBridge:
    """Serial communication with the ESP32."""
    def __init__(self, port, baud=SERIAL_BAUD):
        try:
            import serial
        except ImportError:
            log.error("pyserial not installed. Install with: pip install pyserial")
            sys.exit(1)
        
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # aspetta che ESP32 faccia il boot
        # Svuota il buffer
        self.ser.reset_input_buffer()
        log.info(f"Seriale aperta: {port} @ {baud} baud")
    
    def send_features(self, features):
        """Invia 24 float come JSON all'ESP32."""
        msg = json.dumps({"f": [round(f, 4) for f in features]}) + "\n"
        self.ser.write(msg.encode('ascii'))
    
    def read_score(self):
        """Read the score from the ESP32 (non-blocking)."""
        if self.ser.in_waiting > 0:
            try:
                line = self.ser.readline().decode('ascii').strip()
                if line:
                    data = json.loads(line)
                    return data.get('s', None)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        return None
    
    def close(self):
        self.ser.close()


class DashboardServer:
    """WebSocket server for the dashboard."""
    def __init__(self, port=WEBSOCKET_PORT):
        self.port = port
        self.clients = set()
        self.loop = None
        self._thread = None
    
    def start(self):
        """Start the WebSocket server on a separate thread."""
        try:
            import asyncio
            import websockets
        except ImportError:
            log.error("websockets not installed. Install with: pip install websockets")
            sys.exit(1)
        
        def run():
            import asyncio
            import websockets
            
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            async def handler(websocket):
                self.clients.add(websocket)
                remote = websocket.remote_address
                log.info(f"Dashboard connessa: {remote}")
                try:
                    async for _ in websocket:
                        pass  # la dashboard non ci manda nulla
                except Exception:
                    pass
                finally:
                    self.clients.discard(websocket)
                    log.info(f"Dashboard disconnessa: {remote}")
            
            async def serve():
                async with websockets.serve(handler, "0.0.0.0", self.port):
                    log.info(f"WebSocket server listening on ws://localhost:{self.port}")
                    await asyncio.Future()  # run forever
            
            self.loop.run_until_complete(serve())
        
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
    
    def broadcast(self, data_dict):
        """Send data to every connected dashboard."""
        if not self.clients or self.loop is None:
            return
        
        import asyncio
        
        msg = json.dumps(data_dict)
        
        async def _send():
            disconnected = set()
            for ws in self.clients.copy():
                try:
                    await ws.send(msg)
                except Exception:
                    disconnected.add(ws)
            self.clients -= disconnected
        
        asyncio.run_coroutine_threadsafe(_send(), self.loop)


def extract_features_from_state(state):
    """
    Extract the 24 features from the RTDE packet in the correct order.
    
    The `state` object returned by RTDE has attributes such as:
      state.actual_q        → lista di 6 float
      state.actual_current  → lista di 6 float
      state.actual_qd       → lista di 6 float
      state.actual_TCP_force → lista di 6 float
    """
    features = []
    for field_name in FEATURE_NAMES:
        values = getattr(state, field_name, None)
        if values is None:
            log.error(f"Campo RTDE mancante: {field_name}")
            return None
        if isinstance(values, (list, tuple)):
            features.extend(values)
        else:
            features.append(float(values))
    
    if len(features) != N_FEATURES:
        log.error(f"Attese {N_FEATURES} feature, ottenute {len(features)}")
        return None
    
    return features


def run_bridge(args):
    """Loop principale del bridge."""
    
    # --- 1. Initialisation ---
    log.info("=" * 50)
    log.info("UR5 Anomaly Detection — Lab Bridge")
    log.info(f"  Mode: {args.mode}")
    log.info(f"  Robot IP: {args.robot_ip}")
    if args.mode == 'esp32':
        log.info(f"  Serial: {args.serial_port} @ {SERIAL_BAUD}")
    log.info("=" * 50)
    
    # Load the scaling parameters from the JSON next to the selected ONNX.
    # Applies to both modes (pc/esp32): scaling always happens on the PC.
    global SCALER_PARAMS
    SCALER_PARAMS = load_scaler_params(resolve_scaler_path(args))

    # Buffer circolare
    ring = RingBuffer(WINDOW_SIZE, N_FEATURES)

    # Inference engine
    inference_engine = None
    serial_bridge = None
    
    if args.mode == 'pc':
        if not args.onnx_model:
            log.error("--mode pc richiede --onnx-model "
                      "(es. models/ICE/combined/mlp_autoencoder.onnx)")
            sys.exit(1)
        onnx_path = args.onnx_model
        if not os.path.exists(onnx_path):
            log.error(f"ONNX model not found: {onnx_path}")
            sys.exit(1)
        inference_engine = OnnxInferenceEngine(onnx_path)
    elif args.mode == 'esp32':
        if not args.serial_port:
            log.error("Specificare --serial-port (es. COM3 su Windows)")
            sys.exit(1)
        serial_bridge = SerialBridge(args.serial_port)
    
    # Dashboard WebSocket server
    dashboard = DashboardServer(WEBSOCKET_PORT)
    dashboard.start()
    
    # --- 2. Connessione RTDE ---
    import rtde.rtde as rtde_lib
    import rtde.rtde_config as rtde_config
    
    config_file = os.path.join(PROJECT_ROOT, 'anomaly_config_24.xml')
    if not os.path.exists(config_file):
        log.error(f"RTDE config not found: {config_file}")
        sys.exit(1)
    
    conf = rtde_config.ConfigFile(config_file)
    output_names, output_types = conf.get_recipe("out")
    
    def connect_rtde():
        """(Re)establish the RTDE connection and start streaming. Returns `con` or raises."""
        con = rtde_lib.RTDE(args.robot_ip, RTDE_PORT)
        con.connect()
        con.get_controller_version()
        # NB: ALWAYS use the original names (not the recorder-augmented ones)
        if not con.send_output_setup(output_names, output_types, frequency=FREQUENCY):
            con.disconnect()
            raise RuntimeError("Impossibile configurare output RTDE. Controlla anomaly_config_24.xml.")
        if not con.send_start():
            con.disconnect()
            raise RuntimeError("Impossibile avviare lo stream RTDE.")
        return con

    # --- RECORDER configuration (uses collect.py) ---
    # Names kept separate from the RTDE ones: the recorder prepends 'anomaly_score',
    # but send_output_setup must receive ONLY fields that really exist on the robot.
    recorder = None
    if args.record:
        if StateRecorder is None:
            log.error("Impossibile importare StateRecorder da collect.py")
        else:
            rec_conf = rtde_config.ConfigFile(os.path.join(PROJECT_ROOT, "anomaly_config_24.xml"))
            rec_names, rec_types = rec_conf.get_recipe("out")
            rec_names.insert(0, "anomaly_score")
            rec_types.insert(0, "DOUBLE")
            recorder = StateRecorder(args.record, rec_names, rec_types)
            log.info(f"Registrazione attiva su: {args.record}")

    # --- 3. Loop principale con AUTO-RECONNECT ---
    # The UR controller closes RTDE when it stops (protective/safety stop, e-stop,
    # stop from the pendant): in that case we do NOT exit, we keep the WebSocket
    # alive and retry until the robot becomes available again.
    RECONNECT_DELAY = 3.0  # secondi tra i tentativi di riconnessione
    last_score = 0.0
    sample_count = 0
    dashboard_interval = max(1, FREQUENCY // DASHBOARD_FPS)
    inference_times = deque(maxlen=100)
    con = None
    stop = False

    try:
        while not stop:
            # --- (Ri)connessione ---
            try:
                log.info(f"Connecting to UR5 {args.robot_ip}:{RTDE_PORT}...")
                con = connect_rtde()
                log.info("RTDE stream started! Waiting for data...")
                dashboard.broadcast({"status": "connected"})
            except KeyboardInterrupt:
                stop = True
                break
            except Exception as e:
                log.warning(f"Connessione RTDE non riuscita: {e}. Nuovo tentativo tra {RECONNECT_DELAY:.0f}s...")
                dashboard.broadcast({"status": "reconnecting"})
                time.sleep(RECONNECT_DELAY)
                continue

            # --- Streaming ---
            try:
                while not stop:
                    state = con.receive()
                    if state is None:
                        continue

                    features = extract_features_from_state(state)
                    if features is None:
                        continue

                    # Normalise the features for the model (same parameters as training)
                    features_scaled = scale_features(features)

                    # The model buffer receives the SCALED data
                    ring.push(features_scaled)
                    sample_count += 1

                    # --- Inferenza ---
                    if ring.ready and (sample_count % INFERENCE_STEP == 0):
                        t0 = time.perf_counter()

                        if args.mode == 'pc':
                            last_score = inference_engine.predict(ring.get_window_flat())

                        elif args.mode == 'esp32':
                            # In ESP32 mode we stream every sample and read the
                            # score back when the ESP32 produces it
                            pass

                        t1 = time.perf_counter()
                        inference_ms = (t1 - t0) * 1000
                        inference_times.append(inference_ms)

                        if sample_count % (FREQUENCY * 2) == 0:
                            avg_ms = sum(inference_times) / len(inference_times)
                            log.info(f"Score: {last_score:.5f} | Inferenza: {avg_ms:.1f}ms | Campioni: {sample_count}")

                    # --- ESP32: invio streaming (dati SCALATI) ---
                    if args.mode == 'esp32' and serial_bridge:
                        serial_bridge.send_features(features_scaled)
                        score_from_esp = serial_bridge.read_score()
                        if score_from_esp is not None:
                            last_score = score_from_esp

                    # --- Dashboard update (RAW data, for readability) ---
                    if sample_count % dashboard_interval == 0:
                        dashboard.broadcast({
                            "s": round(last_score, 6),
                            "f": [round(f, 4) for f in features],  # RAW: unità fisiche reali
                        })

                    # --- Registrazione CSV (tramite collect.py) ---
                    if recorder:
                        # Inject the score into the state object before saving it
                        setattr(state, "anomaly_score", float(last_score))
                        recorder.write(state)

            except KeyboardInterrupt:
                stop = True
            except Exception as e:
                # RTDE drop (typically "received 0 bytes": the controller has
                # stopped). We do not exit: close, reset and retry.
                log.error(f"RTDE connection lost: {e}")
                log.warning("The controller has stopped (likely a protective/safety stop). "
                            f"Automatic re-attach in {RECONNECT_DELAY:.0f}s "
                            "(release the stop from the teach pendant if needed)...")
                dashboard.broadcast({"status": "reconnecting"})
                try:
                    con.disconnect()
                except Exception:
                    pass
                con = None
                # Buffer reset: avoids windows straddling the stop (false positives)
                ring = RingBuffer(WINDOW_SIZE, N_FEATURES)
                sample_count = 0
                last_score = 0.0
                time.sleep(RECONNECT_DELAY)

    except KeyboardInterrupt:
        log.info("\nInterrotto dall'utente.")
    finally:
        log.info("Pulizia...")
        if con is not None:
            try:
                con.send_pause()
            except Exception:
                pass
            try:
                con.disconnect()
            except Exception:
                pass
        if serial_bridge:
            serial_bridge.close()
        if recorder:
            recorder.close()
            log.info(f"File {args.record} saved successfully.")
        log.info("Disconnected. Bye!")


def main():
    parser = argparse.ArgumentParser(
        description='UR5 Anomaly Detection — Lab Bridge',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Inference on the PC (no ESP32 needed)
  python bridge.py --robot-ip 192.168.1.100 --mode pc

  # Inference on the ESP32 over serial
  python bridge.py --robot-ip 192.168.1.100 --mode esp32 --serial-port COM3

  # Specify a different ONNX model
  python bridge.py --robot-ip 192.168.1.100 --mode pc --onnx-model models/onnx/cnn_autoencoder.onnx
  
  # Record the dataset using the collect.py recorder
  python bridge.py --robot-ip 192.168.1.100 --mode pc --record dataset_anomalie.csv
        """
    )
    
    parser.add_argument('--robot-ip', required=True, help='IP address of the UR5')
    parser.add_argument('--mode', choices=['pc', 'esp32'], required=True, help='Where to run inference')
    parser.add_argument('--serial-port', default='COM3', help='ESP32 serial port (only for mode=esp32)')
    parser.add_argument('--onnx-model', type=str, default=None,
                        help='Path to the ONNX model. Needed only with --mode pc '
                             '(in esp32 mode the board runs the inference).')
    parser.add_argument('--scaler', type=str, default=None,
                        help='Path to scaler.json. ALWAYS needed (the bridge scales the '
                             'features before sending them). If omitted it is looked up next '
                             'to --onnx-model. Example: models/ICE/combined/scaler.json')
    parser.add_argument('--record', type=str, default=None,
                        help='CSV file to record the RTDE stream and the scores (via collect.py)')
    
    args = parser.parse_args()
    run_bridge(args)


if __name__ == '__main__':
    main()
