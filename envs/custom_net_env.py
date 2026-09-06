"""
Custom OSM-based traffic network environment.
Dynamically builds neighbor map and phase map from the SUMO net.xml file.
"""
import configparser
import logging
import numpy as np
import os
import xml.etree.cElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.env import PhaseMap, PhaseSet, TrafficSimulator
from custom_net.data.build_file import gen_rou_file, get_tls_ids, NET_FILE


STATE_NAMES = ['wave', 'wait', 'ev']

# Actual TLS junction IDs discovered from map.osm
TLS_IDS = [
    '5460773146',
    '5528559489',
    '5528559490',
    'cluster_13067792373_13067792374_1645179383',
    'cluster_2212827519_2212827520_268789820',
    'cluster_2212827553_246824741',
]


def build_neighbor_map(net_path, tls_ids, radius=1):
    """
    Build neighbor map from net.xml by finding TLS junctions
    connected by a direct edge within `radius` hops.
    """
    tls_set = set(tls_ids)
    tree = ET.ElementTree(file=net_path)

    # Map edge -> (from_junction, to_junction)
    edge_map = {}
    for edge in tree.getroot().findall('edge'):
        eid = edge.get('id', '')
        if not eid.startswith(':'):
            edge_map[eid] = (edge.get('from', ''), edge.get('to', ''))

    # Direct neighbors: TLS junctions reachable in 1 hop
    neighbor_map = {tid: [] for tid in tls_ids}
    for eid, (frm, to) in edge_map.items():
        if frm in tls_set and to in tls_set and frm != to:
            if to not in neighbor_map[frm]:
                neighbor_map[frm].append(to)
            if frm not in neighbor_map[to]:
                neighbor_map[to].append(frm)

    return neighbor_map


def build_phase_map(net_path, tls_ids):
    """
    Build phase map by reading tlLogic entries from net.xml.
    Returns a PhaseMap and a dict mapping tls_id -> phase_key.
    """
    tree = ET.ElementTree(file=net_path)
    phase_map = PhaseMap()
    phase_node_map = {}

    for tl in tree.getroot().findall('tlLogic'):
        tl_id = tl.get('id')
        if tl_id not in tls_ids:
            continue
        phases = []
        for phase in tl.findall('phase'):
            state = phase.get('state', '')
            # Only keep green/red phases (skip yellow-only)
            if any(c in state for c in 'Gg'):
                phases.append(state)
        # Must have at least 2 phases for the policy network
        if len(phases) < 2:
            # Pad with a complementary phase or use generic fallback
            if len(phases) == 1:
                # Create complement: swap G<->r
                complement = phases[0].replace('G', 'X').replace('r', 'G').replace('X', 'r')
                phases.append(complement)
            else:
                phases = ['GGrr', 'rrGG']
        key = tl_id
        phase_map.phases[key] = PhaseSet(phases)
        phase_node_map[tl_id] = key

    return phase_map, phase_node_map


class CustomNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='',
                 is_record=False, record_stat=False):
        self.flow_rate = config.getint('flow_rate')
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map.get(node_name, node_name)

    def _init_map(self):
        net_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'custom_net', 'data', 'in', NET_FILE
        )
        net_path = os.path.normpath(net_path)

        if not os.path.exists(net_path):
            raise FileNotFoundError(
                'Network file not found: %s\n'
                'Run: python custom_net/data/build_file.py first.' % net_path
            )

        tls_ids = get_tls_ids(net_path)
        if not tls_ids:
            raise ValueError('No traffic lights found in %s' % net_path)

        self.neighbor_map = build_neighbor_map(net_path, tls_ids)
        self.phase_map, self.phase_node_map = build_phase_map(net_path, tls_ids)
        self.state_names = STATE_NAMES
        # Use lane API (like real_net) instead of lanearea detectors
        # so ilds_in from TraCI always resolve without needing add file detectors
        self.name = 'custom_net'

        logging.info('CustomNetEnv: %d TLS junctions loaded' % len(tls_ids))

    def _init_sim_config(self, seed):
        data_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'custom_net', 'data'
        ))
        return gen_rou_file(
            data_path,
            flow_rate=self.flow_rate,
            seed=seed,
            thread=self.sim_thread
        )

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            sorted_data = np.sort(data)
            yvals = np.arange(len(sorted_data)) / float(max(len(sorted_data) - 1, 1))
            plt.plot(sorted_data, yvals)
            plt.ylabel(name)
            fig.savefig(self.output_path + 'custom_net_' + name + '.png')
            plt.close(fig)


class CustomNetController:
    """
    Greedy wave-maximising controller for the custom OSM network.
    Reads phase strings dynamically from env.phase_map (built from net.xml),
    so it works with any topology without hardcoded NODES/PHASES dicts.
    Includes EV priority override and random tie-breaking.
    """
    def __init__(self, node_names, env):
        self.name = 'greedy'
        self.node_names = node_names
        self.env = env

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        node = self.env.nodes[node_name]
        phase_id = node.phase_id          # e.g. '5460773146'
        phase_set = self.env.phase_map.phases.get(phase_id)

        if phase_set is None:
            return 0  # no phase info — hold phase 0

        phases = phase_set.phases         # list of phase strings e.g. ['GGGrrrr', 'rrrGGGr', ...]
        n_actions = len(phases)
        ilds_in = node.ilds_in            # ordered list of incoming lane/detector IDs

        # ── EV priority override ─────────────────────────────────────────────
        for lane in node.lanes_in:
            try:
                vehicle_ids = self.env.sim.lane.getLastStepVehicleIDs(lane)
                for vid in vehicle_ids:
                    vtype = self.env.sim.vehicle.getTypeID(vid)
                    if 'ambulance' in vtype.lower() or 'emergency' in vtype.lower():
                        # find which phase gives green to this lane
                        try:
                            lane_idx = list(node.lanes_in).index(lane)
                        except ValueError:
                            continue
                        for action_idx, phase_str in enumerate(phases):
                            if lane_idx < len(phase_str) and phase_str[lane_idx] in 'Gg':
                                logging.debug(
                                    '[EV PRIORITY] %s: ambulance %s on lane %s → phase %d',
                                    node_name, vid, lane, action_idx
                                )
                                return action_idx
            except Exception:
                pass

        # ── Greedy wave maximisation ─────────────────────────────────────────
        # For each action, sum the vehicle counts on all lanes that get Green
        flows = np.zeros(n_actions)
        for action_idx, phase_str in enumerate(phases):
            wave = 0.0
            counted = set()
            for lane_idx, signal in enumerate(phase_str):
                if signal in 'Gg' and lane_idx < len(ilds_in):
                    ild = ilds_in[lane_idx]
                    if ild not in counted:
                        j = ilds_in.index(ild)
                        # ob layout: wave states first (len = len(ilds_in))
                        if j < len(ob):
                            wave += ob[j]
                        counted.add(ild)
            flows[action_idx] = wave

        # random tie-breaking to prevent lane starvation
        max_flow = np.max(flows)
        max_indices = np.where(flows == max_flow)[0]
        return int(np.random.choice(max_indices))

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    config = configparser.ConfigParser()
    config.read('./config/config_ma2c_custom.ini')
    env = CustomNetEnv(config['ENV_CONFIG'], port=0,
                       output_path='./output/', is_record=True)
    logging.info('Nodes: %s' % env.node_names)
    logging.info('State dims: %s' % env.n_s_ls)
    env.terminate()
