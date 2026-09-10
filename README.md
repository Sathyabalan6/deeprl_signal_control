# 🚦 Multi-Agent Deep Reinforcement Learning for Traffic Signal Control

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![SUMO Simulator](https://img.shields.io/badge/SUMO_Simulator-Microscopic_Traffic-000000?style=for-the-badge&logo=sumo&logoColor=white)](https://eclipse.dev/sumo/)
[![TensorFlow / PyTorch](https://img.shields.io/badge/Framework-TensorFlow_/_PyTorch-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent **Multi-Agent Deep Reinforcement Learning (MARL)** system designed to dynamically control traffic signal timing across multi-intersection networks simulated via **Eclipse SUMO**. Evaluated on real-world map data imported from **OpenStreetMap (Chennai, India)** with priority override for **Emergency Vehicles (Ambulances)**.

---

## 🌟 Key Highlights

- 🧠 **Multi-Agent Advantage Actor-Critic (MA2C)**: Implements policy fingerprinting and spatial neighborhood coordination for decentralized signal control.
- 🗺️ **Real OpenStreetMap Integration**: Simulates a 6-intersection road network in Guindy/Saidapet/Little Mount (Chennai, India) constructed from raw `.osm` data.
- 🚨 **Emergency Vehicle Priority**: Incorporates shaped reward functions and lane sensor state representations to detect and expedite ambulances through green-wave overrides.
- 📊 **Benchmarking & Baselines**: Supports IA2C, IQLD, IQLL, and traditional greedy wave signal controllers.

---

## 🏗️ System Architecture

```
OpenStreetMap (map.osm)
        │
        ▼
   netconvert (SUMO tool) ──► custom.net.xml + custom.add.xml (Detectors)
        │
        ▼
  SUMO Simulator ◄──────────── TraCI TCP Protocol ────────────► Python (envs/env.py)
                                                                       │
                                                                       ▼
                                                             MA2C Policy Network
                                                             (Actor-Critic + LSTM)
```

---

## 📁 Repository Structure

```
deeprl_signal_control/
├── main.py                     # Entry point for training and evaluation
├── utils.py                    # Trainer, Tester, and Evaluator routines
├── agents/
│   ├── models.py               # MA2C, IA2C, and IQL model architectures
│   └── policies.py             # FPLstmACPolicy (LSTM + Fingerprinting networks)
├── envs/
│   ├── env.py                  # Base TrafficSimulator class interacting via TraCI
│   └── custom_net_env.py       # Environment wrapper for Chennai road network
├── custom_net/data/            # SUMO network configuration, routes, and OSM data
└── config/                     # Hyperparameter INI files (MA2C, IA2C, IQL)
```

---

## 🚀 Quickstart & Setup

### Prerequisites
1. **Python 3.10+**
2. **Eclipse SUMO** installed and added to your system `PATH`:
   ```bash
   sumo --version
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/Sathyabalan6/deeprl_signal_control.git
cd deeprl_signal_control

# Install dependencies
pip install -r requirements.txt
```

### Running Training

```bash
python main.py --base-dir ./output/ma2c train \
    --config-dir ./config/config_ma2c_custom.ini \
    --test-mode no_test
```

### Running Evaluation / Demo with SUMO GUI

```bash
python main.py --base-dir ./output evaluate \
    --agents ma2c \
    --evaluation-seeds 10000 \
    --demo
```

---

## 📊 Results & Visualization

Monitor training logs, state values, and reward metrics with TensorBoard:

```bash
tensorboard --logdir=./output/ma2c/log
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
