import subprocess, time, traci
from sumolib import checkBinary

# Valid TLS-touching edges confirmed from check_net.py
EDGES = [
    '206819132#2', '1015894852#1', '22922908#1', '211260242#12',
    '1031127132#1', '22922908#11', '186477184#2', '22688331#5',
    '24730725#1', '1078273882#2', '1078273881#1', '1082485668',
    '186477184#3', '22688331#7', '22922908#2', '23634681#0',
    '24775612#1', '24775613', '211260242#1', '211260242#13',
    '1015894852#2', '319828651#0', '793704832#0',
]

cmd = [checkBinary('sumo'), '-c', 'custom_net/data/custom_0.sumocfg',
       '--remote-port', '8098', '--no-step-log', 'True', '--no-warnings', 'True']
subprocess.Popen(cmd)
time.sleep(3)
sim = traci.connect(port=8098)

valid_pairs = []
for src in EDGES:
    for dst in EDGES:
        if src == dst:
            continue
        route = sim.simulation.findRoute(src, dst)
        if route.edges:
            valid_pairs.append((src, dst))

sim.close()
print('Valid OD pairs (%d):' % len(valid_pairs))
for s, d in valid_pairs[:30]:
    print(' ', s, '->', d)
