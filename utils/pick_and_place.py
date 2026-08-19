"""
UR5 Pick & Place - lab task runner
=================================================
Runs a repetitive pick-and-place cycle on the UR5.

Positions are stored per task in waypoints_task{N}.json: switching between Task 1
and Task 2 (different pick/place stations) is done with --task, both in teach and
in run mode.

USAGE:
  1. TEACH MODE: move the robot by hand and save the positions of the chosen task
     python pick_and_place.py --robot-ip 10.197.12.12 --teach --task 1

  2. RUN MODE: runs the cycle in a loop using the positions of the chosen task
     python pick_and_place.py --robot-ip <ip_address> --run --task 1
     python pick_and_place.py --robot-ip <ip_address> --run --task 2

  3. RUN together with the bridge (in two separate terminals):
     Terminale 1: python pick_and_place.py --robot-ip <ip_address> --run --task 1
     Terminale 2: python models/deploy/bridge.py --robot-ip <ip_address> --mode pc
"""

import sys
import os
import socket
import time
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RTDE_Python_Client_Library'))
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

# --- CONFIG ---
RTDE_PORT = 30004
COMMAND_PORT = 30003      # Real-time client interface (per inviare URScript)
DASHBOARD_PORT = 29999    # Dashboard server (per stop/play/pause)


def waypoints_path(task):
    """Path of the waypoints file for the chosen task (e.g. waypoints_task1.json)."""
    return os.path.join(os.path.dirname(__file__), f'waypoints_task{task}.json')

# Speed and acceleration (CONSERVATIVE, for lab safety)
VELOCITY = 0.4      # rad/s (lento e sicuro)
ACCELERATION = 0.4  # rad/s²
BLEND_RADIUS = 0.01 # m

# Gripper: digital output index (adatta al tuo setup)
GRIPPER_DO = 0  # Digital Output 0 per apri/chiudi gripper


def send_urscript(ip, script):
    """Invia un comando URScript al robot via socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    sock.connect((ip, COMMAND_PORT))
    sock.sendall((script + "\n").encode('utf-8'))
    time.sleep(0.1)
    sock.close()


def get_current_joints(ip):
    """Read the current joint positions via RTDE."""
    config_content = """<?xml version="1.0"?>
<rtde_config>
    <recipe key="out">
        <field name="actual_q" type="VECTOR6D"/>
    </recipe>
</rtde_config>"""
    
    # Scrivi un config temporaneo
    tmp_config = os.path.join(os.path.dirname(__file__), '_tmp_config.xml')
    with open(tmp_config, 'w') as f:
        f.write(config_content)
    
    conf = rtde_config.ConfigFile(tmp_config)
    output_names, output_types = conf.get_recipe("out")
    
    con = rtde.RTDE(ip, RTDE_PORT)
    con.connect()
    con.get_controller_version()
    con.send_output_setup(output_names, output_types, frequency=10)
    con.send_start()
    
    state = con.receive()
    joints = list(state.actual_q) if state else None
    
    con.send_pause()
    con.disconnect()
    
    # Cleanup
    try:
        os.remove(tmp_config)
    except:
        pass
    
    return joints


def teach_mode(ip, task):
    """Teach mode: the operator moves the robot and the positions are saved."""

    wp_file = waypoints_path(task)
    print("=" * 55)
    print(f"  UR5 PICK & PLACE — TEACH MODE (Task {task})")
    print("=" * 55)
    print()
    print("  Instructions:")
    print("  1. Enable Freedrive on the teach pendant")
    print("  2. Move the robot to the desired position")
    print("  3. Press ENTER to save the position")
    print("  4. Repeat for every position")
    print()
    
    waypoint_names = [
        ("home",      "HOME position (safe rest position)"),
        ("approach",  "APPROACH position (above the object, ~10 cm higher)"),
        ("pick",      "PICK position (where to grasp the object)"),
        ("place_approach", "PLACE APPROACH position (above the release point)"),
        ("place",     "PLACE position (where to release the object)"),
    ]
    
    waypoints = {}
    
    for name, description in waypoint_names:
        print(f"\n>>> Move the robot to: {description}")
        input(f"    Press ENTER when the robot is in position...")
        
        joints = get_current_joints(ip)
        if joints is None:
            print("    ERROR: cannot read the joint positions. Try again.")
            return
        
        waypoints[name] = joints
        print(f"    ✓ Salvato '{name}': [{', '.join(f'{j:.4f}' for j in joints)}]")
    
    # Save to file
    with open(wp_file, 'w') as f:
        json.dump(waypoints, f, indent=2)

    print(f"\n✅ Waypoints salvati in: {wp_file}")
    print(f"   Ora lancia: python pick_and_place.py --robot-ip {ip} --run --task {task}")


def load_waypoints(task):
    """Load the saved waypoints for the chosen task."""
    wp_file = waypoints_path(task)
    if not os.path.exists(wp_file):
        print(f"ERROR: waypoints file not found: {wp_file}")
        print(f"        Run first: python pick_and_place.py --robot-ip XXX --teach --task {task}")
        sys.exit(1)

    with open(wp_file, 'r') as f:
        return json.load(f)

def send_urscript(ip, script):
    """Invia un programma URScript completo al robot via socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((ip, COMMAND_PORT))
    if not script.endswith("\n"):
        script += "\n"
    sock.sendall(script.encode('utf-8'))
    time.sleep(0.2)
    sock.close()


def fmt_joints(joints):
    """Format a list of 6 joint values for URScript."""
    return "[" + ", ".join(f"{j:.5f}" for j in joints) + "]"


def build_full_program(waypoints, vel, acc, use_gripper, gripper_type, n_cycles, invert_gripper=False):
    """
    Build a complete URScript program that the robot executes internally,
    without interruptions. Each movej blocks until the robot
    non arriva a destinazione → movimenti fluidi e completi.
    """
    home = fmt_joints(waypoints['home'])
    approach = fmt_joints(waypoints['approach'])
    pick = fmt_joints(waypoints['pick'])
    place_app = fmt_joints(waypoints['place_approach'])
    place = fmt_joints(waypoints['place'])

    # Gripper polarity logic (pneumatic)
    # Se invert_gripper=True, scambiamo True e False
    val_open = "True" if invert_gripper else "False"
    val_close = "False" if invert_gripper else "True"

    # Comandi gripper in base al tipo
    if not use_gripper:
        open_grip = "sleep(0.3)"
        close_grip = "sleep(0.3)"
    elif gripper_type == 'robotiq':
        open_grip = "rq_open()\n    sleep(0.8)"
        close_grip = "rq_close()\n    sleep(0.8)"
    elif gripper_type == 'gimatic':
        # Gimatic pneumatico
        open_grip = f"set_tool_digital_out({GRIPPER_DO}, {val_open})\n    sleep(0.5)"
        close_grip = f"set_tool_digital_out({GRIPPER_DO}, {val_close})\n    sleep(0.5)"
    elif gripper_type == 'do':
        open_grip = f"set_digital_out({GRIPPER_DO}, {val_open})\n    sleep(0.5)"
        close_grip = f"set_digital_out({GRIPPER_DO}, {val_close})\n    sleep(0.5)"
    else:
        open_grip = "sleep(0.3)"
        close_grip = "sleep(0.3)"

    loop_cond = f"i < {n_cycles}" if n_cycles > 0 else "True"

    program = f"""def pick_and_place():
  # Velocita e accelerazione
  local v = {vel}
  local a = {acc}
  local v_slow = {vel * 0.4}
  local a_slow = {acc * 0.4}

  # SAFETY: make sure the gripper is closed before moving
  {close_grip}
  
  local i = 0
  while {loop_cond}:
    # ====== ANDATA: Pick -> Place ======

    # HOME
    movej({home}, a=a, v=v)

    # Apri gripper
    {open_grip}

    # APPROACH (sopra oggetto)
    movej({approach}, a=a, v=v)

    # PICK (scendi lento)
    movej({pick}, a=a_slow, v=v_slow)

    # Chiudi gripper (afferra)
    {close_grip}

    # Risali ad APPROACH
    movej({approach}, a=a, v=v)

    # PLACE APPROACH
    movej({place_app}, a=a, v=v)

    # PLACE (scendi lento)
    movej({place}, a=a_slow, v=v_slow)

    # Open gripper (release)
    {open_grip}

    # Move back up to PLACE APPROACH
    movej({place_app}, a=a, v=v)

    # Pause between the outward and return legs
    sleep(1.0)

    # ====== RITORNO: Place -> Pick ======

    # PLACE (scendi per riprendere)
    movej({place}, a=a_slow, v=v_slow)

    # Chiudi gripper (afferra)
    {close_grip}

    # Risali a PLACE APPROACH
    movej({place_app}, a=a, v=v)

    # APPROACH
    movej({approach}, a=a, v=v)

    # PICK (scendi per depositare)
    movej({pick}, a=a_slow, v=v_slow)

    # Apri gripper (rilascio)
    {open_grip}

    # Risali ad APPROACH
    movej({approach}, a=a, v=v)

    i = i + 1
    # Pausa tra cicli
    sleep(2.5)
  end

  # Torna a HOME alla fine
  movej({home}, a=a, v=v)
end
"""
    return program


def run_cycle(ip, waypoints, use_gripper=True, gripper_type='do', cycles=0, invert_gripper=False, task=1):
    """
    Invia un programma URScript completo al robot.
    The robot executes it internally with smooth motions.
    """
    print("=" * 55)
    print(f"  UR5 PICK & PLACE — RUN MODE (Task {task})")
    print("=" * 55)
    print(f"  Robot IP: {ip}")
    print(f"  Gripper: {gripper_type if use_gripper else 'disabled'} (Inverted: {invert_gripper})")
    print(f"  Cycles: {'Infinite' if cycles == 0 else cycles}")
    print(f"  Speed: {VELOCITY} rad/s")
    print()

    program = build_full_program(waypoints, VELOCITY, ACCELERATION,
                                  use_gripper, gripper_type, cycles, invert_gripper)

    print("Sending program to the robot...")
    send_urscript(ip, program)
    print("[OK] Program sent! The robot now moves autonomously.")
    print()
    print("  To stop it:")
    print("  - Press the STOP button on the teach pendant")
    print("  - Or Ctrl+C here (a stop command will be sent)")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInvio comando di STOP al robot...")
        # 1. Frena i giunti via URScript (port 30003)
        try:
            send_urscript(ip, "stopj(2.0)")
        except:
            pass
        # 2. Ferma il programma via Dashboard Server (port 29999)
        try:
            ds = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ds.settimeout(3)
            ds.connect((ip, DASHBOARD_PORT))
            ds.recv(1024)  # welcome message
            ds.sendall(b"stop\n")
            time.sleep(0.3)
            ds.close()
        except:
            pass
        print("Robot fermato.")


def main():
    parser = argparse.ArgumentParser(
        description='UR5 Pick & Place - lab task runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--robot-ip', required=True, help='IP address of the UR5 robot')
    parser.add_argument('--task', type=int, default=1,
                        help='Pick & place task number: selects waypoints_task{N}.json (default: 1)')
    parser.add_argument('--teach', action='store_true', help='Teach mode (save positions)')
    parser.add_argument('--run', action='store_true', help='Run the pick & place cycle')
    parser.add_argument('--cycles', type=int, default=0, help='Number of cycles (0 = infinite)')
    parser.add_argument('--no-gripper', action='store_true', help='Disable gripper commands')
    parser.add_argument('--gripper-type', choices=['do', 'robotiq', 'gimatic', 'none'], default='do',
                        help='Gripper type: do (Digital Output), robotiq, none')
    parser.add_argument('--invert-gripper', action='store_true', 
                        help='Invert the open/close commands (True=Open, False=Close)')
    parser.add_argument('--speed', type=float, default=0.3, help='Speed (rad/s, default 0.3)')
    
    args = parser.parse_args()
    
    global VELOCITY
    VELOCITY = args.speed
    
    if args.teach:
        teach_mode(args.robot_ip, args.task)
    elif args.run:
        wp = load_waypoints(args.task)
        gripper_type = 'none' if args.no_gripper else args.gripper_type

        # By default we assume the gimatic is inverted for this setup
        invert = True if gripper_type == 'gimatic' else args.invert_gripper
        if args.invert_gripper:  # L'utente può comunque forzarlo da command line
            invert = True

        run_cycle(args.robot_ip, wp,
                  use_gripper=(gripper_type != 'none'),
                  gripper_type=gripper_type,
                  cycles=args.cycles,
                  invert_gripper=invert,
                  task=args.task)
    else:
        parser.print_help()
        print("\n⚠ Specifica --teach o --run")


if __name__ == '__main__':
    main()
