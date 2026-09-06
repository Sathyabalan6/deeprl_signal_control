# Deep Reinforcement Learning for Traffic Signal Control
## Project Presentation Guide

---

## 1. What Is This Project?

This project applies **Multi-Agent Deep Reinforcement Learning (MARL)** to control traffic signals at real road intersections. Instead of using fixed-time signal cycles, RL agents learn — through millions of simulated interactions — to dynamically switch signal phases to minimize vehicle queues, waiting times, and emergency vehicle delays.

The custom map used is a **real area in Chennai, India** (Guindy / Saidapet / Little Mount region), imported directly from OpenStreetMap and simulated using the SUMO traffic simulator.

---

## 2. Why Is This Problem Important?

- Fixed-time traffic signals waste green time even when roads are empty
- Adaptive signal control can reduce average waiting time by 20–40%
- Emergency vehicles (ambulances) get stuck at red lights — this system gives them priority
- Manual tuning of signal timings across a city is impractical; RL learns automatically
- Scales to large networks without hand-crafted rules

---

## 3. Project Architecture — Big Picture

```
OpenStreetMap (map.osm)
        |
        v
   netconvert (SUMO tool)
        |
        v
  custom.net.xml  ──────────────────────────────────────────┐
        |                                                    |
        v                                                    v
  build_file.py                                    custom.add.xml
  (generates routes,                               (lane detectors)
   detectors, configs)
        |
        v
  custom_0.sumocfg ──► SUMO Simulator ◄──► TraCI (Python API)
                                                    |
                                                    v
                                          CustomNetEnv (env.py)
                                                    |
                                          ┌─────────┴──────────┐
                                          |                     |
                                     State (s)            Reward (r)
                                          |                     |
                                          v                     |
                                    MA2C Model ◄────────────────┘
                                  (TensorFlow)
                                          |
                                          v
                                    Action (phase)
                                          |
                                          v
                              TraCI: setRedYellowGreenState()
```

---

## 4. Project File Structure

```
deeprl_signal_control/
│
├── main.py                          ← Entry point: train / evaluate
├── utils.py                         ← Trainer, Tester, Evaluator, Counter classes
│
├── agents/
│   ├── models.py                    ← A2C, IA2C, MA2C, IQL model classes
│   ├── policies.py                  ← Neural network architectures (LSTM, DQN, LR)
│   └── utils.py                     ← OnPolicyBuffer, ReplayBuffer, Scheduler, fc/lstm layers
│
├── envs/
│   ├── env.py                       ← Base TrafficSimulator class (state, reward, TraCI)
│   ├── custom_net_env.py            ← Chennai map environment
│   ├── small_grid_env.py            ← 6-intersection benchmark
│   ├── large_grid_env.py            ← 5×5 grid (25 intersections)
│   └── real_net_env.py              ← Monaco network (30 intersections)
│
├── custom_net/
│   └── data/
│       ├── build_file.py            ← Generates all SUMO input files from OSM
│       ├── custom_0.sumocfg         ← SUMO configuration file
│       └── in/
│           ├── map.osm              ← Raw OpenStreetMap data (Chennai)
│           ├── custom.net.xml       ← SUMO road network (converted from OSM)
│           ├── custom.add.xml       ← Lane area detectors
│           └── custom_0.rou.xml     ← Vehicle routes and flows
│
└── config/
    └── config_ma2c_custom.ini       ← All hyperparameters for Chennai map
```

---

## 5. The Custom Map — Chennai, India

**Location:** Guindy / Saidapet / Little Mount area
**Coordinates:** lat 13.005–13.019, lon 80.217–80.242

### 6 Controlled Intersections (Traffic Lights)

| Junction ID | Description |
|---|---|
| `5460773146` | Near Guindy area |
| `5528559489` | Connected to 5528559490 |
| `5528559490` | Connected to 5528559489 (neighbor pair) |
| `cluster_13067792373_...` | Merged cluster junction |
| `cluster_2212827519_...` | Merged cluster junction |
| `cluster_2212827553_...` | Merged cluster junction |

**Code location:** `envs/custom_net_env.py` → `TLS_IDS` list

### How the Map Was Built

```
map.osm  →  netconvert  →  custom.net.xml
```

`custom_net/data/build_file.py` → `convert_osm_to_net()` runs netconvert with flags:
- `--tls.guess-signals` — detects traffic lights from OSM tags
- `--tls.join` — merges nearby signals into cluster junctions
- `--junctions.join` — simplifies complex intersections

---

## 6. The RL Problem Formulation

### State Space (what each agent observes)

Each agent observes 3 types of features for every incoming lane:

| Feature | Description | Code |
|---|---|---|
| `wave` | Vehicle count per lane (normalized by `norm_wave=5`) | `env.py` → `_measure_state_step()` |
| `wait` | Max waiting time per lane (normalized by `norm_wait=100`) | `env.py` → `_measure_state_step()` |
| `ev` | Emergency vehicle presence (0/1) + normalized distance | `env.py` → `_measure_state_step()` |

**State dimensions per agent** (from training log):
- `5460773146`: 16 (4 lanes × 4 features)
- `5528559489`: 8 (1 lane × 8 features)
- `5528559490`: 8
- `cluster_13067...`: 20 (5 lanes)
- `cluster_22128...519`: 24 (6 lanes)
- `cluster_22128...553`: 24 (6 lanes)

**Code:** `envs/env.py` → `_init_state_space()`, `_measure_state_step()`

### Action Space (what each agent decides)

Each agent selects a **signal phase** (which lanes get green light).
The number of phases per junction comes from the `tlLogic` entries in `custom.net.xml`.

**Action dimensions:** `[2, 2, 2, 4, 6, 2]` — one per junction

**Code:** `envs/env.py` → `_init_action_space()`, `envs/custom_net_env.py` → `build_phase_map()`

### Reward Function

```
reward = - queue - coef_wait × wait - ev_penalty
```

- `queue` = number of halting vehicles on incoming lanes
- `wait` = max waiting time on incoming lanes
- `ev_penalty` = waiting_time × `coef_ev` (5.0) for any ambulance vehicle

**Code:** `envs/env.py` → `_measure_reward_step()`

---

## 7. The MA2C Algorithm (Main Algorithm Used)

**MA2C = Multi-Agent Advantage Actor-Critic with Fingerprinting**

### How It Works

Each agent has its own **Actor-Critic neural network** with LSTM memory. What makes MA2C different from independent agents is **fingerprinting** — each agent shares a summary of its own policy (the fingerprint) with its neighbors. This helps agents coordinate without full communication.

```
Agent i observes:
  - Local state (wave + wait + ev for its own lanes)
  - Neighbor wave states (scaled by coop_gamma = 0.9)
  - Neighbor EV states
  - Neighbor fingerprints (previous policy probabilities)
        |
        v
  FPLstmACPolicy (TensorFlow)
  ┌─────────────────────────────────┐
  │  wave → FC(128) ──┐             │
  │  wait → FC(32)  ──┼─► LSTM(64) ─► Actor (π)  → phase probabilities │
  │  ev   → FC(32)  ──┤             │
  │  fp   → FC(64)  ──┘         └─► Critic (V) → state value      │
  └─────────────────────────────────┘
```

### Training Loop

```
1. Reset environment (new episode)
2. For each step:
   a. Forward pass → get policy π and value V
   b. Update fingerprint (share policy with neighbors)
   c. Sample action from π
   d. Execute action in SUMO via TraCI
   e. Observe next state and reward
   f. Store (s, a, r, V, done) in OnPolicyBuffer
3. Every 120 steps (batch_size):
   a. Compute returns R and advantages A = R - V
   b. Backward pass → update weights via RMSProp
4. Repeat for 1,000,000 total steps
```

**Code:** `utils.py` → `Trainer.run()`, `Trainer.explore()`
**Model:** `agents/models.py` → `MA2C` class
**Policy network:** `agents/policies.py` → `FPLstmACPolicy`

### Loss Function

```
L = L_policy + L_value + L_entropy

L_policy  = -E[log π(a|s) × Advantage]   ← learn better actions
L_value   = 0.5 × E[(R - V)²]            ← learn accurate value estimates
L_entropy = -β × E[H(π)]                 ← encourage exploration
```

**Code:** `agents/policies.py` → `ACPolicy.prepare_loss()`

---

## 8. Neural Network Architecture

### For MA2C: FPLstmACPolicy

**File:** `agents/policies.py` → `FPLstmACPolicy`

```
Input: [wave_features | wait_features | ev_features | fingerprint_features]

wave  → Dense(128, ReLU)  ──┐
wait  → Dense(32,  ReLU)  ──┼──► Concat → LSTM(64) → Dense(n_actions, Softmax) = π
ev    → Dense(32,  ReLU)  ──┤                      → Dense(1, Linear)          = V
fp    → Dense(64,  ReLU)  ──┘
```

- Separate π (actor) and V (critic) networks sharing the same input processing
- LSTM maintains temporal memory across timesteps within an episode
- States reset at episode boundaries (`done=True`)

### Hyperparameters (from config_ma2c_custom.ini)

| Parameter | Value | Meaning |
|---|---|---|
| `num_fw` | 128 | FC units for wave features |
| `num_ft` | 32 | FC units for wait/EV features |
| `num_fp` | 64 | FC units for fingerprint |
| `num_lstm` | 64 | LSTM hidden units |
| `lr_init` | 5e-4 | Initial learning rate |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 120 | Steps per update |
| `value_coef` | 0.5 | Weight of value loss |
| `entropy_coef_init` | 0.01 | Entropy regularization |
| `reward_norm` | 500.0 | Reward normalization factor |

---

## 9. Emergency Vehicle Priority

This is a key feature added on top of the base paper.

### How It Works End-to-End

1. **Route file** (`custom_0.rou.xml`) — 3 ambulances defined with `type="ambulance"` and `vClass="emergency"`
2. **Detection** (`env.py` → `_measure_state_step()`) — TraCI checks each lane for vehicles with type containing "ambulance" or "emergency", records presence (0/1) and normalized distance to intersection
3. **Reward shaping** (`env.py` → `_measure_reward_step()`) — extra penalty `waiting_time × coef_ev (5.0)` applied when an EV is waiting
4. **State input** — EV presence and distance fed into the neural network so the agent learns to associate EV presence with high penalty
5. **Result** — agent learns to give green to EV lanes faster to avoid the penalty

**Code path:**
- Detection: `envs/env.py` lines in `_measure_state_step()` — EV section
- Penalty: `envs/env.py` lines in `_measure_reward_step()` — ev_penalty section
- Network input: `agents/policies.py` → `LstmACPolicy._build_net()` — `h2 = fc(ob[:, n_s+n_w:], 'fcev', ...)`

---

## 10. How Python Connects to SUMO (TraCI)

```
Python (env.py)  ←──── TCP socket (port 8000) ────►  SUMO process
```

**Startup sequence** (`env.py` → `_init_sim()`):
```python
subprocess.Popen(['sumo', '-c', sumocfg, '--remote-port', '8000'])
time.sleep(2)
self.sim = traci.connect(port=8000)
```

**Key TraCI calls used:**

| Call | Purpose | Code location |
|---|---|---|
| `sim.trafficlight.getIDList()` | Get all TLS junction IDs | `env.py` → `_init_nodes()` |
| `sim.trafficlight.getControlledLanes()` | Get incoming lanes per junction | `env.py` → `_init_nodes()` |
| `sim.trafficlight.setRedYellowGreenState()` | Set signal phase | `env.py` → `_set_phase()` |
| `sim.simulationStep()` | Advance simulation 1 second | `env.py` → `_simulate()` |
| `sim.lane.getLastStepVehicleNumber()` | Vehicle count (wave state) | `env.py` → `_measure_state_step()` |
| `sim.lane.getLastStepVehicleIDs()` | Vehicle IDs for waiting time | `env.py` → `_measure_state_step()` |
| `sim.vehicle.getWaitingTime()` | Waiting time per vehicle | `env.py` → `_measure_reward_step()` |
| `sim.vehicle.getTypeID()` | Detect ambulance vehicles | `env.py` → `_measure_state_step()` |
| `sim.vehicle.getLanePosition()` | EV distance to intersection | `env.py` → `_measure_state_step()` |

---

## 11. Signal Phase Control

Each control step is **5 seconds** (`control_interval_sec=5`).
Yellow transition is **2 seconds** (`yellow_interval_sec=2`).

**Phase switching sequence** (`env.py` → `step()`):
```
1. Set yellow phase for 2 seconds  → _set_phase(action, 'yellow', 2)
2. Simulate 2 seconds              → _simulate(2)
3. Set green phase for 3 seconds   → _set_phase(action, 'green', 3)
4. Simulate 3 seconds              → _simulate(3)
5. Measure state and reward
```

Yellow phases are automatically computed by comparing previous and current green phases — lanes switching from green to red get yellow, lanes switching from red to green stay red during yellow.

**Code:** `env.py` → `_get_node_phase()`

---

## 12. Training Configuration

**File:** `config/config_ma2c_custom.ini`

```ini
[ENV_CONFIG]
scenario         = custom_net      ← uses Chennai map
agent            = ma2c            ← Multi-Agent A2C
flow_rate        = 300             ← vehicles/hour base rate
episode_length_sec = 3600          ← 1 hour per episode
control_interval_sec = 5           ← decision every 5 seconds
coef_ev          = 5.0             ← ambulance penalty weight
objective        = hybrid          ← minimize queue + wait

[TRAIN_CONFIG]
total_step       = 1000000         ← 1 million training steps
log_interval     = 10000           ← log every 10k steps
```

---

## 13. Traffic Flows (Route File)

**File:** `custom_net/data/in/custom_0.rou.xml`

20 traffic flows covering the full 1-hour simulation with a realistic demand pattern:

| Time Period | Volume | Pattern |
|---|---|---|
| 0–600s | 75 veh/hr | Low (off-peak) |
| 600–1200s | 150 veh/hr | Medium |
| 1200–2400s | 225 veh/hr | Peak hour |
| 2400–3000s | 150 veh/hr | Medium |
| 3000–3600s | 75 veh/hr | Low (off-peak) |

3 ambulances injected at t=100s, t=600s, t=1200s on different routes.

**Code:** `custom_net/data/build_file.py` → `STABLE_FLOWS`, `STABLE_AMBULANCES`, `output_stable_rou()`

---

## 14. How to Run

### Train
```bash
cd e:\cloned_project\deeprl_signal_control
python main.py --base-dir ./output/ma2c train \
    --config-dir ./config/config_ma2c_custom.ini \
    --test-mode no_test
```

### Monitor with TensorBoard
```bash
tensorboard --logdir=./output/ma2c/log
# Open http://localhost:6006
```

### Evaluate
```bash
python main.py --base-dir ./output evaluate \
    --agents ma2c \
    --evaluation-seeds 10000,20000,30000
```

### Demo with SUMO GUI (visual)
```bash
python main.py --base-dir ./output evaluate \
    --agents ma2c \
    --evaluation-seeds 10000 \
    --demo
```

### Quick test (reduced steps)
Change `total_step = 1e4` in `config/config_ma2c_custom.ini`, then run training.

---

## 15. Output Files

After training, results are saved to `./output/ma2c/`:

| File | Contents |
|---|---|
| `log/*.log` | Training logs with step-by-step rewards |
| `model/checkpoint-*` | Saved TensorFlow model weights |
| `data/train_reward.csv` | Reward per episode during training |
| `eva_data/custom_net_ma2c_traffic.csv` | Per-step traffic metrics (queue, speed, wait) |
| `eva_data/custom_net_ma2c_trip.csv` | Per-vehicle trip data including EV trips |
| `eva_data/custom_net_ma2c_control.csv` | Per-step actions taken by each agent |

---

## 16. Algorithms Available

| Agent flag | Algorithm | Key idea |
|---|---|---|
| `ma2c` | Multi-Agent A2C + Fingerprinting | Agents share policy summaries with neighbors |
| `ia2c` | Independent A2C | Each agent acts independently, no coordination |
| `iqld` | Independent Q-Learning (Deep DQN) | Off-policy, experience replay |
| `iqll` | Independent Q-Learning (Linear) | Linear Q-function, fastest to train |
| `greedy` | Greedy wave controller | Baseline: always green to most congested lane |

**Code:** `agents/models.py` → `MA2C`, `IA2C`, `IQL` classes

---

## 17. Likely Panel Questions and Answers

**Q: Why use Reinforcement Learning instead of optimization?**
A: Optimization methods like MPC require a mathematical model of traffic dynamics, which is hard to build accurately. RL learns directly from simulation experience without needing an explicit model. It also adapts to changing traffic patterns automatically.

**Q: What is the difference between MA2C and IA2C?**
A: In IA2C, each agent only sees its own local state and acts independently. In MA2C, agents additionally receive their neighbors' wave states and policy fingerprints. The fingerprint (previous policy probabilities) helps agents anticipate what neighbors will do, enabling coordination without direct communication. Code: `agents/models.py` → `MA2C.__init__()` passes `n_f_ls` (fingerprint sizes) while `IA2C` does not.

**Q: What is a fingerprint?**
A: A fingerprint is the agent's previous action probability distribution (policy), with the last element removed to avoid redundancy. It is shared with neighboring agents as part of their state input. This allows agents to model each other's behavior. Code: `envs/env.py` → `update_fingerprint()`, `agents/models.py` → `node.num_fingerprint = node.n_a - 1`.

**Q: Why LSTM instead of a simple feedforward network?**
A: Traffic state is partially observable and has temporal dependencies — a queue building up over several steps is more informative than a single snapshot. LSTM maintains a hidden state across timesteps within an episode, allowing the agent to use recent history. Code: `agents/policies.py` → `LstmACPolicy._build_net()`.

**Q: How does the emergency vehicle priority work?**
A: The EV presence and distance are part of the state vector fed to the neural network. The reward function applies an extra penalty (`waiting_time × 5.0`) whenever an ambulance is waiting. Through training, the agent learns that having an ambulance wait is very costly and adjusts its policy to give green to EV lanes faster. Code: `envs/env.py` → `_measure_state_step()` (EV detection) and `_measure_reward_step()` (ev_penalty).

**Q: How is the real Chennai map used?**
A: The raw OpenStreetMap `.osm` file is converted to a SUMO road network using `netconvert`. The tool extracts road geometry, lane counts, speed limits, and traffic signal locations. The resulting `custom.net.xml` is then parsed by Python to extract junction IDs, phase plans, and neighbor relationships. Code: `custom_net/data/build_file.py` → `convert_osm_to_net()`, `envs/custom_net_env.py` → `_init_map()`.

**Q: What is the reward function?**
A: `reward = -queue - 0.2 × wait - ev_penalty`. Queue is the number of stopped vehicles, wait is the maximum waiting time on incoming lanes, and ev_penalty is the ambulance waiting time multiplied by 5.0. The negative sign means the agent maximizes reward by minimizing congestion. Code: `envs/env.py` → `_measure_reward_step()`.

**Q: How does the agent control the traffic light?**
A: At each 5-second control interval, the agent selects a phase index (e.g., 0 or 1 for a 2-phase junction). The environment first applies a 2-second yellow transition, then sets the chosen green phase for the remaining 3 seconds using TraCI's `setRedYellowGreenState()`. Code: `envs/env.py` → `step()`, `_set_phase()`, `_get_node_phase()`.

**Q: What is the state dimension and why is it different per agent?**
A: Each agent's state size depends on how many incoming lanes its junction has, plus neighbor information. For example, `cluster_2212827519` has 6 incoming lanes, so its state includes 6 wave values + 6 wait values + 12 EV values (presence + distance) = 24 local features, plus neighbor wave states and fingerprints. Code: `envs/env.py` → `_init_state_space()`.

**Q: How do you prevent the model from overfitting to one traffic pattern?**
A: The random seed is incremented after each episode (`self.seed += 1` in `env.py` → `reset()`), which causes `gen_rou_file` to generate slightly different vehicle departure positions each episode. Evaluation uses completely different seeds (`test_seeds = 10000, 20000, 30000`) never seen during training.

**Q: What is the advantage function?**
A: `Advantage = Return - Value = R - V(s)`. The return R is the discounted sum of future rewards. The value V(s) is the critic's estimate of expected return from state s. The advantage tells the actor whether the action taken was better or worse than average — positive advantage means reinforce this action, negative means discourage it. Code: `agents/utils.py` → `OnPolicyBuffer._add_R_Adv()`.

---

## 18. Key Technical Decisions

| Decision | Reason |
|---|---|
| SUMO simulator | Open-source, realistic microscopic traffic simulation, supports TraCI API |
| LSTM over FC | Traffic has temporal dependencies; LSTM captures queue buildup over time |
| On-policy (A2C) over off-policy (DQN) | More stable for multi-agent settings; no replay buffer staleness |
| Fingerprinting over full communication | Scalable — communication cost doesn't grow with network size |
| Real OSM map | Demonstrates applicability to real-world infrastructure |
| Hybrid reward (queue + wait) | Queue alone misses long-waiting vehicles; wait alone misses total congestion |

---

## 19. Limitations and Future Work

- Only 6 intersections — larger networks need more training time
- SUMO simulation may not perfectly match real traffic behavior
- Ambulance routes are predefined — real EVs have dynamic routes
- No pedestrian or cyclist modeling
- Could extend to multi-intersection coordination with graph neural networks
