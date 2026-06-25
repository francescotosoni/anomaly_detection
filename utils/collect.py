import sys
import time
import logging

# Add the path of the downloaded library to Python modules
sys.path.append("RTDE_Python_Client_Library")
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config
import rtde.csv_writer as csv_writer

# --- CONFIGURATION ---
ROBOT_IP = "192.168.1.100"  # IP address of the robot (ICE Lab)
PORT = 30004                # Standard port for UR RTDE
RECORD_TIME = 15            # Recording duration in seconds
FREQUENCY = 125             # Real frequency (125Hz for CB3, 500Hz for e-Series)

CONFIG_FILE = "anomaly_config.xml"
OUTPUT_FILE = "ur5_telemetry_normal.csv"

def main():
    print(f"Attempting to connect to robot {ROBOT_IP}...")
    
    # 1. Load configuration (field names and types)
    conf = rtde_config.ConfigFile(CONFIG_FILE)
    output_names, output_types = conf.get_recipe("out")

    # 2. Connect to the robot's RTDE
    con = rtde.RTDE(ROBOT_IP, PORT)
    con.connect()

    con.get_controller_version()

    # 3. Setup output frequency and variables
    if not con.send_output_setup(output_names, output_types, frequency=FREQUENCY):
        logging.error("Unable to configure output. Check the XML.")
        sys.exit()

    try:
        # Open a CSV writer to save data directly without in-memory arrays
        with open(OUTPUT_FILE, "w", newline="") as csvfile:
            writer = csv_writer.CSVWriter(csvfile, output_names, output_types)
            writer.writeheader()
            
            # Start RTDE stream
            if not con.send_start():
                logging.error("Unable to start the RTDE stream.")
                sys.exit()

            print("Connection established successfully!")
            print(f"Recording started. Expected duration: {RECORD_TIME} seconds...")
            
            start_time = time.time()
            samples_collected = 0
            
            # 4. Time-based recording loop
            while (time.time() - start_time) < RECORD_TIME:
                # receive() is blocking and waits for the packet from the robot at FREQUENCY rate
                state = con.receive()
                
                if state is not None:
                    writer.writerow(state)
                    samples_collected += 1
                else:
                    break
                    
    except KeyboardInterrupt:
        print("\nManual keyboard interruption.")
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you are on the same network as the robot and the IP is correct.")
    finally:
        print("Recording finished. Disconnecting...")
        con.send_pause()
        con.disconnect()
        print(f"File saved successfully: {OUTPUT_FILE}")
        print(f"Total samples recorded: {samples_collected} at {FREQUENCY} Hz")

if __name__ == "__main__":
    main()