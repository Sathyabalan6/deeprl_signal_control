"""
Build SUMO network and route files from a custom OSM map.
Place your .osm file as: custom_net/data/in/map.osm
Then run: python build_file.py
"""
import logging
import numpy as np
import os
import subprocess
import xml.etree.cElementTree as ET

SUMO_HOME = os.environ.get('SUMO_HOME', r'C:\Program Files (x86)\Eclipse\Sumo').rstrip('\\').rstrip('/')
NETCONVERT = os.path.join(SUMO_HOME, 'bin', 'netconvert.exe')
if not os.path.exists(NETCONVERT):
    NETCONVERT = 'netconvert'  # fallback: rely on PATH


OSM_FILE = 'map.osm'
NET_FILE = 'custom.net.xml'
ADD_FILE = 'custom.add.xml'
ILD_OUT  = 'ild.out'


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def convert_osm_to_net(data_path):
    """Convert OSM file to SUMO network using netconvert."""
    osm_path = os.path.join(data_path, 'in', OSM_FILE)
    net_path = os.path.join(data_path, 'in', NET_FILE)
    if not os.path.exists(osm_path):
        raise FileNotFoundError(
            'OSM file not found: %s\n'
            'Please place your .osm file at custom_net/data/in/map.osm' % osm_path
        )
    subprocess.run([
        NETCONVERT,
        '--osm-files', osm_path,
        '--geometry.remove',
        '--roundabouts.guess',
        '--ramps.guess',
        '--junctions.join',
        '--tls.guess-signals',
        '--tls.discard-simple',
        '--tls.join',
        '--output.original-names',
        '--output.street-names',
        '-o', net_path
    ], check=True)
    logging.info('Network converted: %s' % net_path)
    return net_path


def get_tls_ids(net_path):
    """Parse net.xml and return list of traffic light junction IDs."""
    tree = ET.ElementTree(file=net_path)
    tls_ids = []
    for junction in tree.getroot().findall('junction'):
        if junction.get('type') == 'traffic_light':
            tls_ids.append(junction.get('id'))
    logging.info('Found %d traffic light junctions' % len(tls_ids))
    return tls_ids


def get_controlled_lanes(net_path, tls_ids):
    """Return dict mapping tls_id -> list of incoming lane ids.
    Uses junction incLanes attribute which matches exactly what TraCI reports.
    """
    tree = ET.ElementTree(file=net_path)
    tls_set = set(tls_ids)
    junction_incoming = {tid: [] for tid in tls_ids}

    # Build edge -> list of lane ids
    edge_lanes = {}
    for edge in tree.getroot().findall('edge'):
        eid = edge.get('id', '')
        if not eid.startswith(':'):
            edge_lanes[eid] = [lane.get('id') for lane in edge.findall('lane')]

    # Use junction incLanes — exact match to TraCI getControlledLanes
    for junction in tree.getroot().findall('junction'):
        jid = junction.get('id', '')
        if jid not in tls_set:
            continue
        inc_lanes = junction.get('incLanes', '')
        for lane_id in inc_lanes.split():
            # skip internal lanes
            edge_id = '_'.join(lane_id.split('_')[:-1])
            if edge_id.startswith(':'):
                continue
            if lane_id not in junction_incoming[jid]:
                junction_incoming[jid].append(lane_id)
    return junction_incoming


def output_detectors(net_path, tls_ids, data_path):
    """Generate laneAreaDetector additional file for TLS-controlled lanes only."""
    tree = ET.ElementTree(file=net_path)
    root = tree.getroot()
    tls_set = set(tls_ids)

    # Collect only lanes on edges incoming to TLS junctions
    tls_lanes = {}
    for junction in root.findall('junction'):
        if junction.get('id') not in tls_set:
            continue
        for lane_id in junction.get('incLanes', '').split():
            edge_id = '_'.join(lane_id.split('_')[:-1])
            if edge_id.startswith(':'):
                continue
            tls_lanes[lane_id] = None  # length filled below

    # Get lane lengths
    for edge in root.findall('edge'):
        eid = edge.get('id', '')
        if eid.startswith(':'):
            continue
        for lane in edge.findall('lane'):
            lid = lane.get('id', '')
            if lid in tls_lanes:
                try:
                    tls_lanes[lid] = float(lane.get('length', 50.0))
                except ValueError:
                    tls_lanes[lid] = 50.0

    ild_str = '  <laneAreaDetector file="%s" freq="1" id="%s" lane="%s" pos="%.2f" endPos="-0.1"/>\n'
    content = '<additional>\n'
    for lane_id, lane_len in tls_lanes.items():
        if lane_len is None or lane_len < 1.0:  # skip lanes too short for a detector
            logging.warning('Skipping detector for short lane %s (%.2fm)' % (lane_id, lane_len or 0))
            continue
        det_pos = -min(50.0, lane_len - 0.5)
        content += ild_str % (ILD_OUT, lane_id, lane_id, det_pos)
    content += '</additional>\n'
    add_path = os.path.join(data_path, 'in', ADD_FILE)
    write_file(add_path, content)
    logging.info('Detectors written: %s (%d lanes)' % (add_path, len(tls_lanes)))
    return add_path


def output_flows(flow_rate, seed=None):
    """Generate random OD flows across the network."""
    if seed is not None:
        np.random.seed(seed)
    flow_str = (
        '  <flow id="f_%s" departPos="random_free" from="%s" to="%s" '
        'begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
    )
    output = '<routes>\n'
    output += '  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1" color="1,0,0"/>\n'
    output += (
        '  <vType id="ambulance" vClass="emergency" length="6" accel="7" '
        'decel="12" maxSpeed="50" speedFactor="2.0" color="1,0,0" '
        'guiShape="emergency" lcStrategic="1000" lcCooperative="0.0"/>\n'
    )
    output += '</routes>\n'
    return output


def get_edge_ids(net_path):
    """Return list of non-internal edge IDs that allow passenger vehicles."""
    tree = ET.ElementTree(file=net_path)
    edges = []
    for edge in tree.getroot().findall('edge'):
        eid = edge.get('id', '')
        if eid.startswith(':'):
            continue
        # Skip explicitly pedestrian/rail/bicycle-only edges
        lanes = edge.findall('lane')
        if not lanes:
            continue
        for lane in lanes:
            disallow = lane.get('disallow', '')
            allow = lane.get('allow', '')
            speed = float(lane.get('speed', '0') or '0')
            # Accept lane if: speed > 1 m/s AND not rail/tram/ship only
            if speed > 1.0 and 'rail' not in disallow and 'tram' not in disallow:
                if allow == '' or 'passenger' in allow or 'motor' in allow:
                    edges.append(eid)
                    break
    return edges


def output_flows_with_edges(net_path, flow_rate, seed=None):
    """Generate OD flows using actual edge IDs from the network."""
    if seed is not None:
        np.random.seed(seed)
    edges = get_edge_ids(net_path)
    if len(edges) < 2:
        logging.warning('Not enough edges to generate flows')
        return output_flows(flow_rate, seed)

    flow_str = (
        '  <flow id="f_%d" departPos="random_free" from="%s" to="%s" '
        'begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
    )
    vols = [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0, 0]
    times = list(range(0, 3601, 300))

    output = '<routes>\n'
    output += '  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1" color="1,0,0"/>\n'
    output += (
        '  <vType id="ambulance" vClass="emergency" length="6" accel="7" '
        'decel="12" maxSpeed="50" speedFactor="2.0" color="1,0,0" '
        'guiShape="emergency" lcStrategic="1000" lcCooperative="0.0"/>\n'
    )

    k = 0
    for i in range(len(times) - 1):
        vol = vols[i] if i < len(vols) else 0
        if vol == 0:
            continue
        t_begin, t_end = times[i], times[i + 1]
        # generate random OD pairs proportional to volume
        n_flows = max(1, vol)
        for _ in range(n_flows):
            src, dst = np.random.choice(edges, 2, replace=False)
            output += flow_str % (k, src, dst, t_begin, t_end,
                                  int(flow_rate * vol / 4))
            k += 1
    output += '</routes>\n'
    return output


def inject_ambulances(rou_path, net_path):
    """Inject 9 ambulance vehicles spread across the 3600s episode."""
    edges = get_edge_ids(net_path)
    if len(edges) < 6:
        logging.warning('Not enough edges to inject ambulances')
        return
    tree = ET.ElementTree(file=rou_path)
    root = tree.getroot()
    np.random.seed(42)
    # 9 ambulances, spaced out roughly every 350-400 seconds
    for i in range(9):
        src, dst = np.random.choice(edges, 2, replace=False)
        depart = 100.0 + (i * 350.0)
        ev = ET.Element('vehicle', id='ev_%d' % i, type='ambulance',
                        depart='%.2f' % depart)
        ET.SubElement(ev, 'route', edges='%s %s' % (src, dst))
        root.append(ev)
    # pretty format isn't default in basic ET, but SUMO handles it fine
    tree.write(rou_path)
    logging.info('Spread 9 ambulances securely into %s' % rou_path)


def output_config(data_path, thread=None):
    """Generate SUMO config file."""
    if thread is None:
        rou_file = 'custom.rou.xml'
    else:
        rou_file = 'custom_%d.rou.xml' % int(thread)
    cfg = '<configuration>\n  <input>\n'
    cfg += '    <net-file value="in/%s"/>\n' % NET_FILE
    cfg += '    <route-files value="in/%s"/>\n' % rou_file
    cfg += '    <additional-files value="in/%s"/>\n' % ADD_FILE
    cfg += '  </input>\n  <processing>\n'
    cfg += '    <ignore-route-errors value="true"/>\n'
    cfg += '  </processing>\n  <report>\n'
    cfg += '    <no-warnings value="true"/>\n'
    cfg += '  </report>\n  <time>\n'
    cfg += '    <begin value="0"/>\n    <end value="3600"/>\n'
    cfg += '  </time>\n</configuration>\n'
    return cfg


# Stable hardcoded flows using verified edge IDs from custom.net.xml
# Edges confirmed as ALL (open to cars) and touching TLS junctions
# Minimal background traffic — just enough to create queues at intersections
# Reduced from 20 flows to 3, and volume from 225 to 40 veh/hr
STABLE_FLOWS = [
    ('206819132#2',  '22688331#5',  0, 600, 40),
    ('22922908#1',   '186477184#2', 0, 600, 40),
    ('1031127132#1', '22688331#5',  0, 600, 40),
]

# Looping ambulances sorted strictly by depart time (SUMO requirement)
# 3 corridors x 5 spawns = 15 ambulances, staggered every 40s
STABLE_AMBULANCES = [
    (10.00,  '206819132#2 22688331#5',   'ev_a0'),
    (50.00,  '22922908#1 186477184#2',   'ev_b0'),
    (90.00,  '1078273882#2 1078273881#1','ev_c0'),
    (130.00, '206819132#2 22688331#5',   'ev_a1'),
    (170.00, '22922908#1 186477184#2',   'ev_b1'),
    (210.00, '1078273882#2 1078273881#1','ev_c1'),
    (250.00, '206819132#2 22688331#5',   'ev_a2'),
    (290.00, '22922908#1 186477184#2',   'ev_b2'),
    (330.00, '1078273882#2 1078273881#1','ev_c2'),
    (370.00, '206819132#2 22688331#5',   'ev_a3'),
    (410.00, '22922908#1 186477184#2',   'ev_b3'),
    (450.00, '1078273882#2 1078273881#1','ev_c3'),
    (490.00, '206819132#2 22688331#5',   'ev_a4'),
    (530.00, '22922908#1 186477184#2',   'ev_b4'),
    (570.00, '1078273882#2 1078273881#1','ev_c4'),
]


def output_stable_rou(flow_rate_scale=1.0):
    """Generate route file using stable hardcoded flows."""
    output = '<routes>\n'
    output += '  <vType id="car" length="5" accel="5" decel="10" speedDev="0.1" color="1,0,0"/>\n'
    output += (
        '  <vType id="ambulance" vClass="emergency" length="6" accel="7" '
        'decel="12" maxSpeed="50" speedFactor="2.0" color="1,0,0" '
        'guiShape="emergency" lcStrategic="1000" lcCooperative="0.0"/>\n'
    )
    for i, (src, dst, t0, t1, vph) in enumerate(STABLE_FLOWS):
        scaled = max(1, int(vph * flow_rate_scale))
        output += (
            '  <flow id="f_%d" departPos="random_free" from="%s" to="%s" '
            'begin="%d" end="%d" vehsPerHour="%d" type="car"/>\n'
            % (i, src, dst, t0, t1, scaled)
        )
    # Vehicles must be sorted by depart time among all elements;
    # place them at the end with departs beyond the last flow begin (3000s)
    for depart, edges, vid in STABLE_AMBULANCES:
        output += (
            '  <vehicle id="%s" type="ambulance" depart="%.2f">'
            '<route edges="%s"/></vehicle>\n' % (vid, depart, edges)
        )
    output += '</routes>\n'
    return output


def gen_rou_file(data_path, flow_rate, seed=None, thread=None):
    """Generate route file and sumocfg, return sumocfg path."""
    if thread is None:
        rou_file = 'custom.rou.xml'
    else:
        rou_file = 'custom_%d.rou.xml' % int(thread)

    net_path = os.path.join(data_path, 'in', NET_FILE)
    rou_path = os.path.join(data_path, 'in', rou_file)

    # Only regenerate detectors if the add file does not exist yet
    add_path = os.path.join(data_path, 'in', ADD_FILE)
    if not os.path.exists(add_path):
        tls_ids = get_tls_ids(net_path)
        output_detectors(net_path, tls_ids, data_path)

    # Route file is static (fixed ambulance schedule) — only write once
    if not os.path.exists(rou_path):
        content = output_flows_with_edges(net_path, flow_rate=flow_rate, seed=seed)
        write_file(rou_path, content)
        inject_ambulances(rou_path, net_path)

    # Only write sumocfg once — it never changes between episodes
    cfg_file = 'custom_%d.sumocfg' % thread if thread is not None else 'custom.sumocfg'
    sumocfg_path = os.path.join(data_path, cfg_file)
    if not os.path.exists(sumocfg_path):
        write_file(sumocfg_path, output_config(data_path, thread=thread))
    return sumocfg_path


def main():
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        level=logging.INFO
    )
    data_path = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Convert OSM to SUMO net
    net_path = convert_osm_to_net(data_path)

    # Step 2: Get traffic light IDs
    tls_ids = get_tls_ids(net_path)
    if not tls_ids:
        logging.error('No traffic lights found in network. '
                      'Check your OSM file or add --tls.guess to netconvert.')
        return

    # Step 3: Generate detectors
    output_detectors(net_path, tls_ids, data_path)

    # Step 4: Generate initial route file
    rou_path = os.path.join(data_path, 'in', 'custom.rou.xml')
    write_file(rou_path, output_flows_with_edges(net_path, flow_rate=300, seed=42))
    inject_ambulances(rou_path, net_path)

    # Step 5: Generate sumocfg
    write_file(os.path.join(data_path, 'custom.sumocfg'),
               output_config(data_path))

    logging.info('Done! Found %d signalized intersections.' % len(tls_ids))
    logging.info('TLS IDs: %s' % tls_ids[:10])
    logging.info('Next step: update NEIGHBOR_MAP in envs/custom_net_env.py')


if __name__ == '__main__':
    main()
