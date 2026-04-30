import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

# Define colors (cool and professional)
colors = ["#3498db", "#e74c3c", "#2ecc71"]
sns.set_palette(sns.color_palette(colors))

# Load data
df = pd.read_csv('../predictions/predictions_metrics.csv')

# Calculate mean metrics per class
class_means = df.groupby('Class')[['Dice', 'IoU', 'Sensitivity']].mean().reset_index()

# Melt dataframe for seaborn
df_melted = pd.melt(class_means, id_vars=['Class'], value_vars=['Dice', 'IoU', 'Sensitivity'], 
                    var_name='Metric', value_name='Score')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Grouped barplot
bar_plot = sns.barplot(
    data=df_melted, 
    x='Metric', 
    y='Score', 
    hue='Class',
    edgecolor='black',
    linewidth=1.2,
    alpha=0.9,
    ax=ax
)

# Add annotations to bars
for p in bar_plot.patches:
    height = p.get_height()
    if pd.notnull(height) and height > 0:
        ax.annotate(f'{height:.2f}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold',
                    color='#333333')

# Customize plot
plt.title("Mean Performance Metrics by Tumor Class across 10 Test Patients", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Score (0 to 1)", fontsize=13, fontweight='bold')
plt.xlabel("Evaluation Metric", fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
plt.legend(title='Tumor Sub-region', title_fontsize='12', fontsize='11', loc='upper right', frameon=True, shadow=True)

# Grid and spine tweaks
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.grid(False)
sns.despine(left=True, bottom=False)

# Save
os.makedirs('../figures', exist_ok=True)
plt.savefig('../figures/task2_metrics_results.png', dpi=300, bbox_inches='tight')
print("Successfully generated ../figures/task2_metrics_results.png")
