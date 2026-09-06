import sys, os, time, subprocess, traci, logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
sys.path.insert(0, '.')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from custom_net.data.build_file import gen_rou_file
from sumolib import checkBinary

port = 8001
cfg = gen_rou_file('custom_net/data', flow_rate=50, seed=42, thread=0)
print('cfg:', cfg, flush=True)

cmd = [checkBinary('sumo'), '-c', cfg, '--seed', '42', '--remote-port', str(port),
       '--no-step-log', 'True', '--time-to-teleport', '600',
       '--no-warnings', 'True', '--duration-log.disable', 'True',
       '--ignore-route-errors', 'True']

print('--- Episode 1 (probe port) ---', flush=True)
p = subprocess.Popen(cmd)
time.sleep(4)
sim = traci.connect(port=port)
print('Connected ep1', flush=True)
sim.simulationStep()
sim.close()
p.kill()
p.wait()
time.sleep(2)
print('Killed ep1', flush=True)

print('--- Episode 2 (training port, same port) ---', flush=True)
p2 = subprocess.Popen(cmd)
for i in range(10):
    time.sleep(2)
    rc = p2.poll()
    if rc is not None:
        print('SUMO crashed immediately with exit code %d' % rc, flush=True)
        break
    try:
        sim2 = traci.connect(port=port)
        print('Connected ep2 on attempt %d' % i, flush=True)
        sim2.simulationStep()
        sim2.close()
        p2.kill()
        p2.wait()
        print('Episode 2 OK', flush=True)
        break
    except Exception as e:
        print('Attempt %d: %s' % (i, e), flush=True)
