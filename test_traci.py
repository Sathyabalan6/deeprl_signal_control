import subprocess, time, traci
from sumolib import checkBinary

cmd = [checkBinary('sumo'), '-c', 'custom_net/data/custom_0.sumocfg',
       '--remote-port', '8099', '--no-step-log', 'True',
       '--no-warnings', 'True', '--time-to-teleport', '300']
subprocess.Popen(cmd)
time.sleep(3)
sim = traci.connect(port=8099)
print('Connected! Running 10 steps...')
for _ in range(10):
    sim.simulationStep()
print('Vehicles on road:', len(sim.vehicle.getIDList()))
sim.close()
print('SUCCESS - custom net works with TraCI')
