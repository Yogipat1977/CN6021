import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create beautiful dark mode graph
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 7), facecolor='#1a1a1d')
ax.set_facecolor('#1a1a1d')

df = pd.read_csv('../predictions/predictions_metrics.csv')
class_means = df.groupby('Class')[['Dice', 'IoU', 'Sensitivity']].mean().reset_index()
df_melted = pd.melt(class_means, id_vars=['Class'], value_vars=['Dice', 'IoU', 'Sensitivity'], 
                    var_name='Metric', value_name='Score')

# Modern neon colors
colors = {'NCR/NET': '#ff0055', 'ED': '#00f2fe', 'ET': '#4facfe'}

# Plot bars with slight offset for grouping
metrics = df_melted['Metric'].unique()
classes = df_melted['Class'].unique()
bar_width = 0.25
index = np.arange(len(metrics))

for i, cls in enumerate(classes):
    cls_data = df_melted[df_melted['Class'] == cls]
    scores = cls_data['Score'].values
    pos = index + (i * bar_width) - bar_width
    
    bars = ax.bar(pos, scores, bar_width, label=cls, color=colors[cls], alpha=0.85, 
                  edgecolor='#ffffff', linewidth=1.5, zorder=3)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 6),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ffffff')

# Customizing the look
ax.set_title('Mean 3D Segmentation Metrics by Tumor Sub-Region', fontsize=18, fontweight='bold', color='#ffffff', pad=25)
ax.set_xlabel('Evaluation Metric', fontsize=14, fontweight='bold', color='#aaaaaa', labelpad=15)
ax.set_ylabel('Performance Score (0 - 1)', fontsize=14, fontweight='bold', color='#aaaaaa', labelpad=15)
ax.set_xticks(index)
ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)

# Sexy grid
ax.yaxis.grid(True, linestyle='--', color='#444444', alpha=0.6, zorder=0)
ax.xaxis.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#555555')
ax.spines['bottom'].set_color('#555555')

# Legend
legend = ax.legend(title='Tumor Class', fontsize=12, title_fontsize=13, loc='upper right', frameon=True, facecolor='#2c2f33', edgecolor='#555555')
plt.setp(legend.get_title(), color='#ffffff')
for text in legend.get_texts():
    text.set_color('#ffffff')

plt.tight_layout()

# Save
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/task2_metrics_results_sexy.png', dpi=300, bbox_inches='tight')
print("Successfully generated sexy graph!")
