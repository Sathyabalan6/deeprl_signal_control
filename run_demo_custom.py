"""
Demo script for the custom OSM-based traffic network with SUMO GUI.
Runs greedy controller with EV priority on the real custom map.
"""
import sys
import time
import configparser
import logging

from envs.custom_net_env import CustomNetEnv, CustomNetController

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    level=logging.INFO
)

# Load config — points to custom_net scenario
config = configparser.ConfigParser()
config.read('./config/config_ma2c_custom.ini')

print("=" * 65)
print("  Custom OSM Network — Traffic Simulation with SUMO GUI")
print("=" * 65)

# Initialize environment
env = CustomNetEnv(config['ENV_CONFIG'], port=0)
print(f"\nEnvironment initialized:")
print(f"  Traffic lights : {len(env.node_names)}")
print(f"  Nodes          : {env.node_names}")
print(f"  State dims     : {env.n_s_ls}")
print(f"  Action dims    : {env.n_a_ls}")

# Initialize CustomNetController — reads phases dynamically from net.xml
controller = CustomNetController(env.node_names, env=env)

# Reset with GUI
print("\nStarting SUMO GUI simulation...")
state = env.reset(gui=True)
print(">>> SWITCH TO THE SUMO GUI AND PRESS PLAY! <<<\n")

step = 0
total_reward = 0
ev_seen = set()

try:
    while True:
        actions = controller.forward(state)
        next_state, reward, done, global_reward = env.step(actions)
        total_reward += global_reward
        step += 1

        # Detect any active EVs on the network
        active_evs = []
        try:
            all_vehicles = env.sim.vehicle.getIDList()
            for vid in all_vehicles:
                try:
                    vtype = env.sim.vehicle.getTypeID(vid)
                    if 'ambulance' in vtype.lower() or 'emergency' in vtype.lower():
                        active_evs.append(vid)
                        ev_seen.add(vid)
                except Exception:
                    pass
        except Exception:
            pass

        # Print step summary every 5 steps
        if step % 5 == 1:
            print(f"\n{'='*65}")
            print(f"STEP {step:4d}  |  Sim time: {env.cur_sec:5d}s  |  "
                  f"Reward: {global_reward:7.2f}  |  Total: {total_reward:8.2f}")
            print(f"{'='*65}")
            for i, node_name in enumerate(env.node_names):
                node = env.nodes[node_name]
                action = actions[i]
                # get phase string for this node
                phase_key = env.phase_map.phases.get(node.phase_id)
                phase_str = ''
                if phase_key is not None:
                    try:
                        phase_str = env.phase_map.get_phase(node.phase_id, action)
                    except Exception:
                        phase_str = 'N/A'

                # EV presence at this node
                ev_here = []
                for ild in node.ilds_in:
                    try:
                        for vid in env.sim.lane.getLastStepVehicleIDs(ild):
                            vtype = env.sim.vehicle.getTypeID(vid)
                            if 'ambulance' in vtype.lower():
                                ev_here.append(vid)
                    except Exception:
                        pass

                ev_tag = f"  *** EV: {ev_here} ***" if ev_here else ""
                wave_str = ' '.join(f"{v:.2f}" for v in node.wave_state)
                if node.ev_state is not None and len(node.ev_state) > 0:
                    ev_state_str = ' '.join(str(int(v)) for v in node.ev_state)
                else:
                    ev_state_str = 'n/a'
                print(f"  [{node_name}] action={action} phase={phase_str[:8]:<8}  "
                      f"wave=[{wave_str}]  ev_state=[{ev_state_str}]{ev_tag}")

        if active_evs:
            print(f"  ACTIVE AMBULANCES: {active_evs}")

        if step % 20 == 0:
            print(f"\n  ---> Cumulative EVs seen so far: {sorted(ev_seen)}")

        time.sleep(0.3)  # small pause so GUI stays responsive

        if done:
            print(f"\n{'='*65}")
            print(f"Simulation COMPLETE")
            print(f"  Total steps   : {step}")
            print(f"  Total EVs ran : {len(ev_seen)} — {sorted(ev_seen)}")
            print(f"  Avg reward    : {total_reward / step:.2f}")
            print(f"{'='*65}")
            break

        state = next_state

except KeyboardInterrupt:
    print("\nInterrupted by user.")

env.terminate()
print("Environment terminated.")
