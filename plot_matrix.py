import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

traffic = pd.read_csv('output/eva_data/small_grid_ma2c_traffic.csv')
trip    = pd.read_csv('output/eva_data/small_grid_ma2c_trip.csv')
control = pd.read_csv('output/eva_data/small_grid_ma2c_control.csv')

ev_trips    = trip[trip['is_ev'] == 1]
car_trips   = trip[trip['is_ev'] == 0]

# ── Figure 1: Performance Matrix Table ──────────────────────────────────────
fig1, ax = plt.subplots(figsize=(13, 6))
fig1.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.axis('off')
ax.set_title('MA2C Performance Matrix — Small Grid (6 Intersections)',
             fontsize=14, fontweight='bold', color='white', pad=20)

rows = [
    ['Metric', 'Value', 'Unit', 'Description'],
    ['Algorithm', 'MA2C', '—', 'Multi-Agent Advantage Actor-Critic'],
    ['Network', 'Small Grid', '—', '6 signalized intersections'],
    ['Episode Length', '600', 'seconds', '10-minute simulation'],
    ['Total Vehicles Departed', str(int(trip.shape[0])), 'vehicles', 'All vehicles that entered network'],
    ['Total Vehicles Arrived', str(int(trip[trip['arrival_sec'].notna()].shape[0])), 'vehicles', 'Vehicles that completed trip'],
    ['Trip Completion Rate', f"{trip[trip['arrival_sec'].notna()].shape[0]/trip.shape[0]*100:.1f}%", '—', 'Percentage of trips completed'],
    ['Avg Trip Duration', f"{car_trips['duration_sec'].mean():.1f}", 'seconds', 'Mean travel time per vehicle'],
    ['Avg Vehicle Wait Time', f"{car_trips['wait_sec'].mean():.2f}", 'seconds', 'Mean waiting time at signals'],
    ['Max Vehicle Wait Time', f"{car_trips['wait_sec'].max():.1f}", 'seconds', 'Worst-case waiting time'],
    ['Avg Queue per Lane', f"{traffic['avg_queue'].mean():.3f}", 'vehicles', 'Mean halting vehicles per lane'],
    ['Peak Queue per Lane', f"{traffic['avg_queue'].max():.3f}", 'vehicles', 'Maximum queue observed'],
    ['Avg Vehicle Speed', f"{traffic['avg_speed_mps'].mean():.3f}", 'm/s', 'Mean speed across all vehicles'],
    ['Ambulances in Network', str(len(ev_trips)), 'vehicles', 'Emergency vehicles simulated'],
    ['Avg Ambulance Wait', f"{ev_trips['wait_sec'].mean():.2f}" if len(ev_trips) > 0 else '0.00', 'seconds', 'Mean EV waiting time at signals'],
    ['Max Ambulance Wait', f"{ev_trips['wait_sec'].max():.1f}" if len(ev_trips) > 0 else '0.0', 'seconds', 'Worst-case EV delay'],
    ['Avg Ambulance Trip Time', f"{ev_trips['duration_sec'].mean():.1f}" if len(ev_trips) > 0 else '—', 'seconds', 'Mean EV journey duration'],
    ['EV Priority Coefficient', '5.0', '—', 'Reward penalty multiplier for EV wait'],
]

col_widths = [0.22, 0.12, 0.10, 0.56]
col_colors_header = ['#e94560', '#e94560', '#e94560', '#e94560']
row_colors = [['#16213e', '#16213e', '#16213e', '#16213e'] if i % 2 == 0
              else ['#0f3460', '#0f3460', '#0f3460', '#0f3460']
              for i in range(len(rows)-1)]

table = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    cellLoc='left',
    loc='center',
    colWidths=col_widths
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor('#e94560')
    cell.set_linewidth(0.5)
    if r == 0:
        cell.set_facecolor('#e94560')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor(row_colors[r-1][c])
        cell.set_text_props(color='white')

plt.tight_layout()
plt.savefig('output/performance_matrix.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print('Saved: output/performance_matrix.png')

# ── Figure 2: Agent Heatmap (like confusion matrix) ─────────────────────────
fig2 = plt.figure(figsize=(16, 10))
fig2.patch.set_facecolor('#1a1a2e')
fig2.suptitle('MA2C Agent Performance Heatmap — Small Grid',
              fontsize=14, fontweight='bold', color='white')

gs = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.35)

agents = ['Agent 0', 'Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5']
time_bins = np.arange(0, 601, 60)
bin_labels = [f'{t}-{t+60}s' for t in time_bins[:-1]]

# Build per-agent action matrix from control CSV
# control has columns: episode, time_sec, step, action, reward
# action is comma-separated e.g. "0,1,0,1,0,1"
action_matrix = np.zeros((6, len(bin_labels)))
count_matrix  = np.zeros((6, len(bin_labels)))

for _, row in control.iterrows():
    t = row['time_sec']
    bin_idx = min(int(t // 60), len(bin_labels)-1)
    try:
        actions = [int(a) for a in str(row['action']).split(',')]
        for ag_idx, act in enumerate(actions[:6]):
            action_matrix[ag_idx, bin_idx] += act
            count_matrix[ag_idx, bin_idx]  += 1
    except:
        pass

# Normalize to get avg action (phase selection rate)
with np.errstate(divide='ignore', invalid='ignore'):
    avg_action = np.where(count_matrix > 0, action_matrix / count_matrix, 0)

# 1. Agent Action Heatmap
ax1 = fig2.add_subplot(gs[0, :2])
im = ax1.imshow(avg_action, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
ax1.set_xticks(range(len(bin_labels)))
ax1.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8, color='white')
ax1.set_yticks(range(6))
ax1.set_yticklabels(agents, color='white')
ax1.set_title('Agent Phase Selection Rate Over Time\n(0=Phase 0, 1=Phase 1)', color='white')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('white')
for i in range(6):
    for j in range(len(bin_labels)):
        ax1.text(j, i, f'{avg_action[i,j]:.2f}', ha='center', va='center',
                 fontsize=7, color='black' if avg_action[i,j] > 0.5 else 'white')
plt.colorbar(im, ax=ax1).ax.yaxis.set_tick_params(color='white')

# 2. Reward per time bin
ax2 = fig2.add_subplot(gs[0, 2])
reward_bins = []
for b in range(len(bin_labels)):
    mask = (control['time_sec'] >= time_bins[b]) & (control['time_sec'] < time_bins[b+1])
    reward_bins.append(control.loc[mask, 'reward'].mean() if mask.sum() > 0 else 0)
colors_r = ['#e94560' if r < 0 else '#00b4d8' for r in reward_bins]
ax2.barh(bin_labels, reward_bins, color=colors_r, edgecolor='white', linewidth=0.3)
ax2.set_title('Avg Reward per Time Bin', color='white')
ax2.tick_params(colors='white')
ax2.set_facecolor('#16213e')
for spine in ax2.spines.values():
    spine.set_edgecolor('white')
ax2.axvline(0, color='white', linewidth=0.5)

# 3. Wait time distribution: EV vs Normal
ax3 = fig2.add_subplot(gs[1, 0])
ax3.set_facecolor('#16213e')
bins = np.linspace(0, car_trips['wait_sec'].quantile(0.95), 30)
ax3.hist(car_trips['wait_sec'], bins=bins, color='#00b4d8', alpha=0.7,
         label=f'Normal ({len(car_trips)})', edgecolor='white', linewidth=0.3)
if len(ev_trips) > 0:
    ax3.axvline(ev_trips['wait_sec'].mean(), color='#e94560', linewidth=2,
                linestyle='--', label=f'EV avg ({ev_trips["wait_sec"].mean():.1f}s)')
ax3.set_title('Wait Time Distribution', color='white')
ax3.set_xlabel('Wait (s)', color='white')
ax3.set_ylabel('Count', color='white')
ax3.tick_params(colors='white')
ax3.legend(facecolor='#16213e', labelcolor='white', fontsize=8)
for spine in ax3.spines.values():
    spine.set_edgecolor('white')

# 4. Trip duration distribution
ax4 = fig2.add_subplot(gs[1, 1])
ax4.set_facecolor('#16213e')
ax4.hist(car_trips['duration_sec'], bins=30, color='#06d6a0', alpha=0.8,
         edgecolor='white', linewidth=0.3)
ax4.axvline(car_trips['duration_sec'].mean(), color='#e94560', linewidth=2,
            linestyle='--', label=f'Mean: {car_trips["duration_sec"].mean():.0f}s')
ax4.set_title('Trip Duration Distribution', color='white')
ax4.set_xlabel('Duration (s)', color='white')
ax4.set_ylabel('Count', color='white')
ax4.tick_params(colors='white')
ax4.legend(facecolor='#16213e', labelcolor='white', fontsize=8)
for spine in ax4.spines.values():
    spine.set_edgecolor('white')

# 5. EV vs Normal summary bar
ax5 = fig2.add_subplot(gs[1, 2])
ax5.set_facecolor('#16213e')
categories = ['Avg Wait\n(s)', 'Avg Duration\n(s)', 'Avg Wait\nSteps']
ev_vals  = [ev_trips['wait_sec'].mean(), ev_trips['duration_sec'].mean(),
            ev_trips['wait_step'].mean()] if len(ev_trips) > 0 else [0, 0, 0]
car_vals = [car_trips['wait_sec'].mean(), car_trips['duration_sec'].mean(),
            car_trips['wait_step'].mean()]
x = np.arange(len(categories))
w = 0.35
ax5.bar(x - w/2, car_vals, w, label='Normal Vehicles', color='#00b4d8', alpha=0.8)
ax5.bar(x + w/2, ev_vals,  w, label='Ambulances',      color='#e94560', alpha=0.8)
ax5.set_xticks(x)
ax5.set_xticklabels(categories, color='white', fontsize=8)
ax5.set_title('EV vs Normal Vehicle Metrics', color='white')
ax5.tick_params(colors='white')
ax5.legend(facecolor='#16213e', labelcolor='white', fontsize=8)
for spine in ax5.spines.values():
    spine.set_edgecolor('white')

plt.savefig('output/agent_heatmap.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
print('Saved: output/agent_heatmap.png')
plt.close('all')
print('\nOpen both files:')
print('  output\\performance_matrix.png')
print('  output\\agent_heatmap.png')
