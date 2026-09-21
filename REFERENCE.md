# Deep RL Traffic Signal Control — Reference Guide

## What This Project Does

Multi-agent deep reinforcement learning for traffic signal control in SUMO-simulated road networks. RL agents learn to dynamically control signal phases at intersections to minimize congestion, waiting time, and emergency vehicle delays.further it can be implemented to real life but that requires a input layer as of now it just in stimulation which takes a input as a tag so to get the details form the real life it need a yolo or any other image recognition algo 

---

## Project St
ructure

```
deeprl_signal_control/
├── agents/              # RL algorithm implementations
│   ├── models.py        # A2C, IA2C, MA2C, IQL model classes
│   ├── policies.py      # Neural network architectures (LSTM, DQN, LR)
│   └── utils.py         # OnPolicyBuffer, ReplayBuffer, Scheduler
├── envs/                # SUMO environment wrappers
│   ├── env.py           # Base TrafficSimulator class
│   ├── small_grid_env.py  # 6-intersection benchmark
│   ├── large_grid_env.py  # 5x5 grid (25 intersections)
│   ├── real_net_env.py    # Monaco network (30 intersections)
│   └── custom_net_env.py  # Custom network support
├── config/              # Hyperparameter .ini files
├── small_grid/data/     # SUMO network files for small grid
├── large_grid/data/     # SUMO network files for large grid
├── real_net/data/       # SUMO network files for Monaco
├── custom_net/data/     # SUMO network files for custom net
├── main.py              # Entry point (train / evaluate)
└── utils.py             # Trainer, Tester, Evaluator, Counter
```

---

## Algorithms

| Agent | Description |
|-------|-------------|
| `ia2c` | Independent A2C — each agent acts independently |
| `ma2c` | Multi-Agent A2C — agents share neighbor fingerprints for coordination |
| `iqld` | Independent Q-Learning (Deep DQN) |
| `iqll` | Independent Q-Learning (Linear) |
| `greedy` | Greedy wave-based controller (baseline) |

---

## Environments

| Scenario | Intersections | Config key |
|----------|--------------|------------|
| Small grid | 6 | `small_grid` |
| Large grid | 25 (5×5) | `large_grid` |
| Monaco (real net) | 30 | `real_net` |
| Custom | variable | `custom_net` |

Before training on `small_grid` or `large_grid`, generate SUMO network files:
```bash
python small_grid/data/build_file.py
python large_grid/data/build_file.py
```

---

## State Space

Each agent observes per-lane features for its intersection (and neighbors in MA2C):

| State | Description |
|-------|-------------|
| `wave` | Vehicle count per lane (normalized) |
| `wait` | Max waiting time per lane (normalized) |
| `ev` | Emergency vehicle presence (0/1) + normalized distance to intersection |

State names are defined per environment, e.g. in `small_grid_env.py`:
```python
STATE_NAMES = ['wave', 'wait', 'ev']
```

---

## Reward

Reward per agent per step (in `env.py → _measure_reward_step()`):

```
reward = - queue - coef_wait * wait - ev_penalty
```

- `queue` — number of halting vehicles on incoming lanes
- `wait` — max waiting time on incoming lanes
- `ev_penalty` — `waiting_time × coef_ev` for any emergency/ambulance vehicle (default `coef_ev = 5.0`)

---

## Emergency Vehicle Priority

Emergency vehicles are supported end-to-end:

1. **SUMO route files** — ambulances defined with `vClass="emergency"` or type containing `"ambulance"`
2. **State detection** (`env.py → _measure_state_step()`) — detects EV presence and distance per lane via TraCI
3. **Reward shaping** — extra penalty applied when an EV is waiting
4. **Greedy override** (`small_grid_env.py → SmallGridController.greedy()`) — hard-overrides action to give green to the lane with an approaching ambulance
5. **Trip tracking** — EV trips flagged with `is_ev=1` in output CSVs

---

## Usage

### 1. Train
```bash
python main.py --base-dir ./output/ma2c train \
    --config-dir ./config/config_ma2c_small.ini \
    --test-mode no_test
```
`--test-mode` options: `no_test`, `in_train_test`, `after_train_test`, `all_test`

### 2. Monitor with TensorBoard
```bash
tensorboard --logdir=./output/ma2c/log
```

### 3. Evaluate
```bash
python main.py --base-dir ./output evaluate \
    --agents ma2c,ia2c,greedy \
    --evaluation-seeds 10000,20000,30000
```
Output CSVs saved to `[base_dir]/eva_data/`.

### 4. Demo (SUMO GUI)
```bash
python main.py --base-dir ./output evaluate \
    --agents ma2c \
    --evaluation-seeds 10000 \
    --demo
```

---

## Config File Structure

Located in `config/config_<agent>_<network>.ini`:

```ini
[ENV_CONFIG]
scenario = small_grid
num_agent = 6
seed = 42
norm_wave = 20
norm_wait = 20
clip_wave = 10
clip_wait = 10
coef_wait = 1.0
coef_ev = 5.0
control_interval_sec = 10
yellow_interval_sec = 2
episode_length_sec = 3600

[MODEL_CONFIG]
batch_size = 20
lr_init = 0.001
gamma = 0.95
num_fw = 20
num_ft = 20
num_lstm = 64

[TRAIN_CONFIG]
total_step = 100000
test_interval = 2000
log_interval = 100
```

---

## How Python Connects to SUMO (TraCI)

```
Python RL Code  <--TraCI API-->  SUMO Simulator
```

Key TraCI calls used in `env.py`:

| Call | Purpose |
|------|---------|
| `traci.start()` | Launch SUMO process |
| `traci.simulationStep()` | Advance simulation 1 second |
| `traci.trafficlight.setRedYellowGreenState()` | Set signal phase |
| `traci.lanearea.getLastStepVehicleNumber()` | Get vehicle count (wave) |
| `traci.lanearea.getLastStepVehicleIDs()` | Get vehicle IDs on lane |
| `traci.vehicle.getWaitingTime()` | Get vehicle waiting time |
| `traci.vehicle.getTypeID()` | Detect emergency vehicles |
| `traci.vehicle.getLanePosition()` | Get EV distance to intersection |
| `traci.close()` | Terminate SUMO |

---

## Requirements

- Python 3.5+
- TensorFlow 1.12.0 (`tensorflow.compat.v1`)
- SUMO >= 1.1.0
- numpy, pandas, seaborn, matplotlib

Install dependencies:
```bash
bash setup_ubuntu.sh   # Ubuntu
bash setup_mac.sh      # macOS
```

---

## Citation

```
@article{chu2019multi,
  title={Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control},
  author={Chu, Tianshu and Wang, Jie and Codec{\`a}, Lara and Li, Zhaojian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2019},
  publisher={IEEE}
}
```
