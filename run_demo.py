"""
Simple script to run the traffic simulation with SUMO GUI
"""
import sys
import time
import configparser
import logging

from envs.small_grid_env import SmallGridEnv, SmallGridController

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    level=logging.INFO)

# Load config
config = configparser.ConfigParser()
config.read('./config/config_test_small.ini')

print("="*60)
print("Running Traffic Simulation with SUMO GUI")
print("="*60)

# Initialize environment
env = SmallGridEnv(config['ENV_CONFIG'], port=0)
print(f"Environment initialized: {len(env.node_names)} traffic lights")

# Initialize greedy controller with environment reference
controller = SmallGridController(env.node_names, env=env)

# Reset environment with GUI enabled
print("\nStarting simulation with GUI...")
state = env.reset(gui=True)
print("\n>>> PLEASE SWITCH TO THE SUMO GUI WINDOW AND PRESS THE PLAY BUTTON! <<<")

# Run simulation
step = 0
total_reward = 0
try:
    while True:
        actions = controller.forward(state)
        next_state, reward, done, global_reward = env.step(actions)
        total_reward += global_reward
        step += 1
        
        # Print detailed per-agent information
        print("\n" + "="*60)
        print(f"STEP {step}  (Simulation Time: {env.cur_sec}s)")
        print("="*60)
        
        for i, node_name in enumerate(env.node_names):
            node = env.nodes[node_name]
            neighbors_str = ", ".join(node.neighbor)
            
            print(f"\nAGENT {node_name} | Neighbours: {neighbors_str}")
            
            # State information
            wave_str = "[" + ", ".join([f"{v:.2f}" for v in node.wave_state]) + "]"
            wait_str = "[" + ", ".join([f"{v:.2f}" for v in node.wait_state]) + "]"
            print(f"  State   : wave={wave_str} wait={wait_str}")
            
            # EV State information
            ev_pres_str = "[" + ", ".join([str(int(v)) for v in node.ev_state]) + "]"
            ev_dist_str = "[" + ", ".join([f"{v:.1f}" for v in node.ev_distance]) + "]"
            print(f"  EV State: ev_presence={ev_pres_str} ev_distance={ev_dist_str}")
            
            # Action information
            action = actions[i]
            phase_str = env.phase_map.get_phase(node.phase_id, action)
            print(f"  Action  : Phase {action} (Phase string: {phase_str})")
            
            # Reward information
            agent_reward = reward[i] if hasattr(reward, '__len__') else reward
            print(f"  Reward  : {agent_reward:.2f}")
            
            # Fingerprint (for MA2C agent)
            if env.agent == 'ma2c' and len(node.fingerprint) > 0:
                fp_str = "[" + ", ".join([f"{v:.3f}" for v in node.fingerprint]) + "]"
                print(f"  Fingerprint sent to: {neighbors_str} -> π={fp_str}")
            
            # Check for EV detection
            ev_detected = False
            detected_evs = []
            for lane in node.lanes_in:
                try:
                    vehicle_ids = env.sim.lane.getLastStepVehicleIDs(lane)
                    for vid in vehicle_ids:
                        vtype = env.sim.vehicle.getTypeID(vid)
                        if 'ambulance' in vtype.lower():
                            ev_detected = True
                            detected_evs.append(vid)
                except:
                    pass
            
            if ev_detected:
                evs_str = ", ".join(detected_evs)
                print(f"  *** EV PRIORITY ACTIVE - Ambulance {evs_str} detected! ***")
            else:
                print(f"  [NO EV DETECTED]")
        
        if step % 10 == 0:
            print(f"\n{'='*60}")
            print(f"Step {step}: Reward = {global_reward:.2f}, Total = {total_reward:.2f}")
            print(f"{'='*60}")
        
        # Add a small buffer to observe the GUI simulation securely
        time.sleep(0.5)
        
        if done:
            print(f"\nSimulation completed!")
            print(f"Total steps: {step}")
            print(f"Average reward: {total_reward/step:.2f}")
            break
        
        state = next_state
        
except KeyboardInterrupt:
    print("\nSimulation interrupted by user")

# Cleanup
env.terminate()
print("Environment terminated")
