import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

FIGURES_DIR = Path("figures")

# ── Configuration ───────────────────────────────────────────────────────────
GPU_CONFIG = 'run_20260507_144824' 
STOP_ITERS = [15, 38, 100] 
COLORS_STAGES = ['#1db992', '#f3a31d', '#ef565a'] 

# ── Data Loading & Recalculation ────────────────────────────────────────────
records = []
for f in glob.glob(f'deepcam_results/{GPU_CONFIG}/detailed_logs/*.json'):
    if '._' not in f:
        with open(f) as fh:
            j = json.load(fh)
            if j["objectives"]["energy_kWh"] < 100:
                records.append(j)

records = sorted(records, key=lambda x: (x['iteration'], x['candidate']))

raw_ious = []
raw_energies = []
cum_energies = []
best_ious = []
iterations = []
current_cum_en = 0
current_best_iou = -np.inf

# Process all records for scatter and step-plot logic[cite: 3]
records = records[:102]
for r in records:
    # Wh conversion as seen in the screenshot
    energy_wh = r['objectives']['energy_kWh'] * 1000
    current_cum_en += energy_wh
    current_best_iou = max(current_best_iou, r['objectives']['iou_validation'])
    
    r['cum_en_recalc'] = current_cum_en
    r['best_iou_recalc'] = current_best_iou
    
    raw_ious.append(r['objectives']['iou_validation'])
    raw_energies.append(current_cum_en)
    cum_energies.append(current_cum_en)
    best_ious.append(current_best_iou)
    iterations.append(r['iteration'])

total_energy = cum_energies[-1]

# ── Gradient Step Plot Logic ────────────────────────────────────────────────
x_steps, y_steps, iter_steps = [], [], []
for i in range(len(cum_energies) - 1):
    # Horizontal/Vertical segments for step effect[cite: 3]
    x_steps.extend([cum_energies[i], cum_energies[i+1], cum_energies[i+1]])
    y_steps.extend([best_ious[i], best_ious[i], best_ious[i+1]])
    iter_steps.extend([iterations[i], iterations[i], iterations[i+1]])

points = np.array([x_steps, y_steps]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# ── Plotting ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 9), facecolor='white')
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(0, 100)

# 1. Background Scatter: All individual results
sc = ax.scatter(raw_energies, raw_ious, c=COLORS_STAGES[-1], 
                alpha=0.2, s=25, zorder=1, edgecolors='none')

# 2. Gradient Step Line[cite: 3]
lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3, zorder=3)
lc.set_array(np.array(iter_steps))
ax.add_collection(lc)

# Colorbar for BO Iterations[cite: 1, 3]
cbar = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('BO Iteration', rotation=90, labelpad=15, fontsize=12)

# 3. Milestones, Vertical Lines, and Text Labels[cite: 1]
for i, stop_it in enumerate(STOP_ITERS):
    target_recs = [r for r in records if r['iteration'] == stop_it]
    if not target_recs: continue
    
    m = target_recs[-1]
    en_val, iou_val = m['cum_en_recalc'], m['best_iou_recalc']
    pct = (en_val / total_energy) * 100
    
    # Vertical dash lines[cite: 1]
    ax.axvline(x=en_val, color=COLORS_STAGES[i], linestyle='--', alpha=0.5, zorder=2)
    
    # Large marker dots[cite: 1]
    label = f"S{i+1}: stop iter {stop_it} - {en_val:.0f} Wh ({pct:.0f}%)"
    ax.plot(en_val, iou_val, 'o', color=COLORS_STAGES[i], markersize=12, 
            label=label, zorder=5, markeredgecolor='white')
    
    # Floating label for IoU value
    ax.text(en_val, iou_val + 0.01, f'{iou_val:.3f}', color=COLORS_STAGES[i], 
            fontweight='bold', ha='center', fontsize=10, zorder=6)

# Aesthetics & Labels
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

plt.title('Evolution of the Pareto Frontier (Convergence Analysis)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Cumulative Campaign Energy (Wh)', fontsize=12)
plt.ylabel('IoU (Validation)', fontsize=12)
plt.xlim(0, total_energy * 1.05)
plt.ylim(min(raw_ious) - 0.02, max(best_ious) + 0.05)

# Legend placement below the plot[cite: 1]
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'pareto_gradient_scatter_final.png', dpi=300, bbox_inches='tight')