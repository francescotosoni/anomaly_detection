import sys
import time
import logging
import argparse
import numpy as np
import os

# Add the path of the downloaded library to Python modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "RTDE_Python_Client_Library"))
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import rtde.csv_writer as csv_writer

# --- CONFIGURATION ---
ROBOT_IP = "192.168.1.100"  # IP address of the robot (ICE Lab)
PORT = 30004                # Standard port for UR RTDE
FREQUENCY = 125             # Real frequency (125Hz for CB3, 500Hz for e-Series)

CONFIG_FILE = "anomaly_config_24.xml"

class StateRecorder:
    """
    Utility class to record RTDE states to CSV cleanly.

    NB: we write with a COMMA separator (standard CSV). The RTDE library defaults
    to a space, like the academic Graabaek et al. dataset: the readers downstream
    (lab_retrain.py) detect the separator automatically, so both formats work.
    i vecchi file space-separated restano leggibili.
    """
    def __init__(self, filename, output_names, output_types, delimiter=","):
        self.filename = filename
        # Create the destination folder if missing (e.g. data/lab/task1/)
        out_dir = os.path.dirname(os.path.abspath(filename))
        os.makedirs(out_dir, exist_ok=True)
        self.f = open(filename, "w", newline="")
        self.writer = csv_writer.CSVWriter(self.f, output_names, output_types, delimiter=delimiter)
        self.writer.writeheader()
        
    def write(self, state):
        self.writer.writerow(state)
        
    def close(self):
        self.f.close()


def extract_features(state):
    """Extract the 24 features from the RTDE state."""
    f = []
    f.extend(state.actual_q)
    f.extend(state.actual_current)
    f.extend(state.actual_qd)
    f.extend(state.actual_TCP_force)
    return f

def main():
    parser = argparse.ArgumentParser(description="UR5 Data Collector")
    parser.add_argument('--robot-ip', default=ROBOT_IP, help="IP address of the UR5")
    parser.add_argument('--time', type=int, default=60, help="Recording duration in seconds")
    parser.add_argument('--output', default="ur5_telemetry_normal.csv",
                        help="Output CSV file")
    parser.add_argument('--compute-score', action='store_true',
                        help="Also compute and record the anomaly score (requires ONNX)")
    parser.add_argument('--onnx-model', default=None,
                        help="Path to the ONNX model (required with --compute-score)")
    args = parser.parse_args()

    print(f"Connecting to robot {args.robot_ip}...")
    
    # 1. Load configuration (field names and types)
    conf = rtde_config.ConfigFile(CONFIG_FILE)
    output_names, output_types = conf.get_recipe("out")

    # 2. Connect to the robot's RTDE
    con = rtde.RTDE(args.robot_ip, PORT)
    con.connect()

    con.get_controller_version()

    # 3. Setup output frequency and variables
    if not con.send_output_setup(output_names, output_types, frequency=FREQUENCY):
        logging.error("Unable to configure output. Check the XML.")
        sys.exit()

    samples_collected = 0
    try:
        # If requested, initialise inference
        model = None
        ring = None
        if args.compute_score:
            try:
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'deploy'))
                from bridge import scale_features, OnnxInferenceEngine, RingBuffer
            except ImportError:
                print("ERROR: cannot import modules from bridge.py. The score will not be computed.")
                sys.exit(1)
            model = OnnxInferenceEngine(args.onnx_model)
            ring = RingBuffer(window_size=50, n_features=24)
            # Add the score field to the CSV header
            output_names.insert(0, "anomaly_score")
            output_types.insert(0, "DOUBLE")

        # Initialise our clean recorder
        recorder = StateRecorder(args.output, output_names, output_types)
        
        # Start RTDE stream
        if not con.send_start():
            logging.error("Unable to start the RTDE stream.")
            sys.exit()

        print("Connection established successfully!")
        print(f"Recording started. Expected duration: {args.time} seconds...")

        start_time = time.time()

        # 4. Time-based recording loop
        while (time.time() - start_time) < args.time:
            # receive() is blocking and waits for the packet from the robot at FREQUENCY rate
            state = con.receive()

            if state is not None:
                if args.compute_score:
                    # Estrai le 24 feature grezze
                    features = extract_features(state)
                    features_scaled = scale_features(features)
                    ring.push(features_scaled)
                    
                    score = 0.0
                    if ring.ready:
                        score = model.predict(ring.get_window_flat())
                    
                    # Inject the score into the state object
                    setattr(state, "anomaly_score", float(score))

                recorder.write(state)
                samples_collected += 1
        print("\nRecording completed.")
        recorder.close()

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
        recorder.close()
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you are on the same network as the robot and the IP is correct.")
    finally:
        print("Recording finished. Disconnecting...")
        con.send_pause()
        con.disconnect()
        print(f"File saved successfully: {args.output}")
        print(f"Total samples recorded: {samples_collected} at {FREQUENCY} Hz")

if __name__ == "__main__":
    main()