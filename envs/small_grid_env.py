"""
Particular class of small traffic network
@author: Tianshu Chu
"""

import configparser
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import seaborn as sns
import time
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from small_grid.data.build_file import gen_rou_file

sns.set_color_codes()

SMALL_GRID_NEIGHBOR_MAP = {'nt1': ['npc', 'nt2', 'nt6'],
                           'nt2': ['nt1', 'nt3'],
                           'nt3': ['npc', 'nt2', 'nt4'],
                           'nt4': ['nt3', 'nt5'],
                           'nt5': ['npc', 'nt4', 'nt6'],
                           'nt6': ['nt1', 'nt5']}

STATE_NAMES = ['wave', 'wait', 'ev']
# map from ild order (alphabeta) to signal order (clockwise from north)
STATE_PHASE_MAP = {'nt1': [0, 1, 2], 'nt2': [0, 1], 'nt3': [0, 1],
                   'nt4': [0, 1], 'nt5': [0, 1], 'nt6': [0, 1]}


class SmallGridPhase(PhaseMap):
    def __init__(self):
        two_phase = ['GGrr', 'rrGG']
        three_phase = ['GGGrrrrrr', 'rrrGGGrrr', 'rrrrrrGGG']
        self.phases = {2: PhaseSet(two_phase), 3: PhaseSet(three_phase)}


class SmallGridController:
    def __init__(self, node_names, env=None):
        self.name = 'greedy'
        self.node_names = node_names
        self.env = env

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        phases = STATE_PHASE_MAP[node_name]
        flows = ob[:len(phases)]
        
        # Emergency vehicle priority
        if self.env is not None:
            node = self.env.nodes[node_name]
            
            # Check all vehicles approaching this intersection
            for lane in node.lanes_in:
                try:
                    vehicle_ids = self.env.sim.lane.getLastStepVehicleIDs(lane)
                    for vid in vehicle_ids:
                        vtype = self.env.sim.vehicle.getTypeID(vid)
                        if 'ambulance' in vtype.lower():
                            # Find which phase gives green to this lane
                            lane_idx = node.lanes_in.index(lane)
                            for phase_action in phases:
                                phase_str = self.env.phase_map.get_phase(node.phase_id, phase_action)
                                if lane_idx < len(phase_str) and phase_str[lane_idx] in 'Gg':
                                    print(f"[EV PRIORITY] {node_name}: Ambulance {vid} on lane {lane}, phase {phase_action}")
                                    return phase_action
                except Exception as e:
                    print(f"Error checking EV: {e}")
        
        # Default: most vehicles (with tie-breaker to avoid starvation)
        max_flow = np.max(flows)
        max_indices = np.where(flows == max_flow)[0]
        chosen_index = np.random.choice(max_indices)
        return phases[chosen_index]


class SmallGridEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        self.num_car_hourly = config.getint('num_extra_car_per_hour')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        if node_name == 'nt1':
            return 3
        return 2

    def _init_map(self):
        self.neighbor_map = SMALL_GRID_NEIGHBOR_MAP
        self.phase_map = SmallGridPhase()
        self.state_names = STATE_NAMES

    def _init_sim_config(self, seed):
        return gen_rou_file(seed=seed,
                            thread=self.sim_thread,
                            path=self.data_path,
                            num_car_hourly=self.num_car_hourly)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    plt.plot(sorted_data, yvals, color=c, label=label)

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO)
    config = configparser.ConfigParser()
    config.read('./config/config_test_small.ini')
    base_dir = './output_result/'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    env = SmallGridEnv(config['ENV_CONFIG'], 2, base_dir, is_record=True, record_stat=True)
    ob = env.reset()
    controller = SmallGridController(env.node_names)
    rewards = []
    while True:
        next_ob, _, done, reward = env.step(controller.forward(ob))
        rewards.append(reward)
        if done:
            break
        ob = next_ob
    env.plot_stat(np.array(rewards))
    logging.info('avg reward: %.2f' % np.mean(rewards))
    env.terminate()
    time.sleep(2)
    env.collect_tripinfo()
    env.output_data()
