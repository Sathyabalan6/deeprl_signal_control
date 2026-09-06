import sys, time, subprocess, traci
sys.path.insert(0, '.')
from custom_net.data.build_file import gen_rou_file

for episode in range(3):
    cfg = gen_rou_file('custom_net/data', flow_rate=50, seed=42+episode, thread=0)
    p = subprocess.Popen(['sumo', '-c', cfg, '--seed', str(42+episode),
                          '--remote-port', '8001', '--no-step-log', 'True',
                          '--time-to-teleport', '600', '--no-warnings', 'True',
                          '--duration-log.disable', 'True', '--ignore-route-errors', 'True'])
    connected = False
    for attempt in range(10):
        time.sleep(2)
        try:
            sim = traci.connect(port=8001)
            connected = True
            break
        except Exception as e:
            print('Episode %d attempt %d: %s' % (episode, attempt, e), flush=True)
    if connected:
        sim.simulationStep()
        sim.close()
        p.kill()
        p.wait()
        time.sleep(1)
        print('Episode %d OK' % episode, flush=True)
    else:
        p.kill()
        print('Episode %d FAILED' % episode, flush=True)
print('All done', flush=True)
