import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

df = pd.read_csv('output/eva_data/small_grid_ma2c_traffic.csv')

fig = plt.figure(figsize=(16, 10))
fig.suptitle('MA2C Traffic Signal Control — Small Grid Performance', fontsize=16, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1. Average Queue over time
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df['time_sec'], df['avg_queue'], color='crimson', linewidth=1.2)
ax1.set_title('Avg Queue per Lane')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Vehicles')
ax1.grid(True, alpha=0.3)

# 2. Average Waiting Time over time
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(df['time_sec'], df['avg_wait_sec'], color='darkorange', linewidth=1.2)
ax2.set_title('Avg Waiting Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Seconds')
ax2.grid(True, alpha=0.3)

# 3. Average Speed over time
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(df['time_sec'], df['avg_speed_mps'], color='steelblue', linewidth=1.2)
ax3.set_title('Avg Vehicle Speed')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('m/s')
ax3.grid(True, alpha=0.3)

# 4. Total vehicles in network
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(df['time_sec'], df['number_total_car'], color='green', linewidth=1.2)
ax4.set_title('Total Vehicles in Network')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Count')
ax4.grid(True, alpha=0.3)

# 5. EV waiting time
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(df['time_sec'], df['avg_ev_wait_sec'], color='purple', linewidth=1.2)
ax5.fill_between(df['time_sec'], df['avg_ev_wait_sec'], alpha=0.2, color='purple')
ax5.set_title('Ambulance Avg Wait Time')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Seconds')
ax5.grid(True, alpha=0.3)

# 6. Summary stats bar chart
ax6 = fig.add_subplot(gs[1, 2])
metrics = ['Avg Queue', 'Avg Wait (s)', 'Avg Speed\n(m/s)', 'EV Wait (s)']
values = [
    df['avg_queue'].mean(),
    df['avg_wait_sec'].mean(),
    df['avg_speed_mps'].mean(),
    df['avg_ev_wait_sec'].mean()
]
colors = ['crimson', 'darkorange', 'steelblue', 'purple']
bars = ax6.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax6.set_title('Overall Averages')
ax6.set_ylabel('Value')
for bar, val in zip(bars, values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

plt.savefig('output/performance_summary.png', dpi=150, bbox_inches='tight')
print('Saved: output/performance_summary.png')

# Print summary table
print('\n========== PERFORMANCE SUMMARY ==========')
print(f'Scenario        : Small Grid (6 intersections)')
print(f'Algorithm       : MA2C (Multi-Agent A2C)')
print(f'Simulation Time : {df["time_sec"].max()} seconds')
print(f'------------------------------------------')
print(f'Avg Queue/Lane  : {df["avg_queue"].mean():.3f} vehicles')
print(f'Max Queue/Lane  : {df["avg_queue"].max():.3f} vehicles')
print(f'Avg Wait Time   : {df["avg_wait_sec"].mean():.3f} seconds')
print(f'Max Wait Time   : {df["avg_wait_sec"].max():.3f} seconds')
print(f'Avg Speed       : {df["avg_speed_mps"].mean():.3f} m/s')
print(f'Avg EV Wait     : {df["avg_ev_wait_sec"].mean():.3f} seconds')
print(f'Total Departed  : {df["number_departed_car"].sum()} vehicles')
print(f'Total Arrived   : {df["number_arrived_car"].sum()} vehicles')
print('==========================================')
