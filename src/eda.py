"""
eda.py — Exploratory Data Analysis for Customer Churn Dataset.

Produces all EDA figures and saves them to figures/ directory.
Uses a minimalistic dark theme with shades of blue and black.

Group: Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for figure saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Plot Theme — Minimalistic dark with shades of blue and black
# ---------------------------------------------------------------------------

# Colour palette
BG_DARK      = '#0D1117'       # GitHub-dark background
BG_PANEL     = '#161B22'       # Panel / axes background
TEXT_PRIMARY  = '#E6EDF3'      # Primary text
TEXT_SECONDARY = '#8B949E'     # Secondary / tick text
BLUE_LIGHT   = '#58A6FF'       # Accent — light blue
BLUE_MID     = '#1F6FEB'       # Accent — mid blue
BLUE_DARK    = '#0D419D'       # Accent — dark blue
BLUE_PALE    = '#79C0FF'       # Highlight
GRID_COLOR   = '#21262D'       # Subtle grid

# Two-class palette (retained=blue, churned=lighter blue)
CLASS_PALETTE = {0: BLUE_DARK, 1: BLUE_LIGHT}
CLASS_LABELS  = {0: 'Retained', 1: 'Churned'}

# Custom blue sequential colormap
BLUE_CMAP = LinearSegmentedColormap.from_list(
    'custom_blue', [BG_DARK, BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE], N=256
)

# Apply global theme
plt.rcParams.update({
    'figure.facecolor':    BG_DARK,
    'axes.facecolor':      BG_PANEL,
    'axes.edgecolor':      GRID_COLOR,
    'axes.labelcolor':     TEXT_PRIMARY,
    'axes.titlesize':      14,
    'axes.titleweight':    'bold',
    'axes.labelsize':      11,
    'text.color':          TEXT_PRIMARY,
    'xtick.color':         TEXT_SECONDARY,
    'ytick.color':         TEXT_SECONDARY,
    'xtick.labelsize':     9,
    'ytick.labelsize':     9,
    'legend.facecolor':    BG_PANEL,
    'legend.edgecolor':    GRID_COLOR,
    'legend.fontsize':     9,
    'legend.labelcolor':   TEXT_PRIMARY,
    'grid.color':          GRID_COLOR,
    'grid.linestyle':      '--',
    'grid.alpha':          0.5,
    'font.family':         'sans-serif',
    'figure.dpi':          150,
    'savefig.dpi':         200,
    'savefig.facecolor':   BG_DARK,
    'savefig.bbox':        'tight',
    'savefig.pad_inches':  0.3,
})


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')


def save_fig(fig, name):
    """Save figure to figures/ directory."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"   Saved: {path}")


# ---------------------------------------------------------------------------
# EDA Functions
# ---------------------------------------------------------------------------

def plot_class_distribution(df):
    """Bar chart of Churn class distribution."""
    counts = df['Churn'].value_counts().sort_index()
    total = len(df)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [CLASS_LABELS[i] for i in counts.index],
        counts.values,
        color=[CLASS_PALETTE[i] for i in counts.index],
        edgecolor=BG_DARK, linewidth=1.5, width=0.5
    )
    # Add count + percentage labels on bars
    for bar, cnt in zip(bars, counts.values):
        pct = cnt / total * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{cnt:,}\n({pct:.1f}%)', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=TEXT_PRIMARY)
    ax.set_title('Class Distribution')
    ax.set_ylabel('Count')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, counts.max() * 1.15)
    save_fig(fig, 'class_distribution.png')


def plot_feature_distributions(df, numerical_cols):
    """Histograms/KDE plots for numerical features, split by churn label."""
    n = len(numerical_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        for label in [0, 1]:
            subset = df[df['Churn'] == label][col].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=CLASS_PALETTE[label],
                    label=CLASS_LABELS[label], density=True, edgecolor='none')
        ax.set_title(col, fontsize=11)
        ax.legend(framealpha=0.8)
        ax.grid(axis='y', alpha=0.2)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Distributions by Churn Status', fontsize=15,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'feature_distributions.png')


def plot_categorical_distributions(df, categorical_cols):
    """Grouped bar charts for categorical features by churn label."""
    n = len(categorical_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, categorical_cols):
        ct = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
        x = np.arange(len(ct))
        w = 0.35
        ax.bar(x - w/2, ct[0], w, label=CLASS_LABELS[0], color=BLUE_DARK,
               edgecolor=BG_DARK, linewidth=0.8)
        ax.bar(x + w/2, ct[1], w, label=CLASS_LABELS[1], color=BLUE_LIGHT,
               edgecolor=BG_DARK, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ct.index, rotation=0)
        ax.set_title(col, fontsize=11)
        ax.set_ylabel('Percentage (%)')
        ax.legend(framealpha=0.8)
        ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Categorical Feature Distributions by Churn', fontsize=15,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'categorical_distributions.png')


def plot_correlation_heatmap(df, numerical_cols, categorical_cols):
    """Pearson correlation matrix heatmap."""
    # Encode categoricals temporarily for correlation
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

    cols = numerical_cols + categorical_cols + ['Churn']
    corr = df_encoded[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap=BLUE_CMAP, center=0,
                annot=True, fmt='.2f', linewidths=0.5,
                linecolor=BG_DARK, ax=ax,
                annot_kws={'size': 8, 'color': TEXT_PRIMARY},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
    # Fix tick label colours
    ax.tick_params(colors=TEXT_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    save_fig(fig, 'correlation_heatmap.png')


def plot_boxplots(df, numerical_cols):
    """Box plots for numerical features to identify outliers."""
    n = len(numerical_cols)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        ax = axes[i]
        data = [df[df['Churn'] == 0][col].dropna(), df[df['Churn'] == 1][col].dropna()]
        bp = ax.boxplot(data, labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
                        patch_artist=True, widths=0.4,
                        medianprops=dict(color=BLUE_PALE, linewidth=2),
                        whiskerprops=dict(color=TEXT_SECONDARY),
                        capprops=dict(color=TEXT_SECONDARY),
                        flierprops=dict(marker='o', markersize=3,
                                        markerfacecolor=BLUE_LIGHT, alpha=0.4))
        for patch, colour in zip(bp['boxes'], [BLUE_DARK, BLUE_MID]):
            patch.set_facecolor(colour)
            patch.set_edgecolor(TEXT_SECONDARY)
            patch.set_alpha(0.8)
        ax.set_title(col, fontsize=11)
        ax.grid(axis='y', alpha=0.2)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Feature Box Plots by Churn Status', fontsize=15,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'boxplots.png')


def plot_missing_values(df):
    """Bar chart of missing values per column."""
    nan_counts = df.isnull().sum()
    nan_pct = nan_counts / len(df) * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(nan_counts.index, nan_counts.values, color=BLUE_MID,
                   edgecolor=BG_DARK, linewidth=0.8)
    # Add count labels
    for bar, cnt, pct in zip(bars, nan_counts.values, nan_pct.values):
        if cnt > 0:
            ax.text(bar.get_width() + max(nan_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{cnt:,} ({pct:.1f}%)', va='center', fontsize=9, color=BLUE_LIGHT)
    ax.set_title('Missing Values per Column', fontsize=14, fontweight='bold')
    ax.set_xlabel('Count')
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    save_fig(fig, 'missing_values.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda():
    """Run complete EDA pipeline."""
    print("=" * 60)
    print("Exploratory Data Analysis")
    print("=" * 60)

    # Import preprocessing
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocessing import download_dataset, load_data

    # Load raw data (before cleaning, so we can visualise NaN)
    print("\n1. Loading raw data...")
    train_path, test_path = download_dataset()
    train_raw, _ = load_data(train_path, test_path)
    print(f"   Shape: {train_raw.shape}")
    print(f"   Columns: {list(train_raw.columns)}")
    print(f"   Dtypes:\n{train_raw.dtypes.to_string()}")

    # Identify column types
    numerical_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                      'Payment Delay', 'Total Spend', 'Last Interaction']
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']

    # Drop CustomerID for analysis
    df = train_raw.drop(columns=['CustomerID'])

    # 2. Missing values
    print("\n2. Missing value analysis...")
    nan_total = df.isnull().sum().sum()
    nan_rows = df.isnull().any(axis=1).sum()
    print(f"   Total NaN values: {nan_total:,}")
    print(f"   Rows with at least one NaN: {nan_rows:,} ({nan_rows/len(df)*100:.2f}%)")
    print(f"   NaN per column:")
    for col in df.columns:
        cnt = df[col].isnull().sum()
        if cnt > 0:
            print(f"     {col}: {cnt:,} ({cnt/len(df)*100:.2f}%)")
    plot_missing_values(df)

    # Clean for remaining plots
    df = df.dropna()

    # 3. Class distribution
    print("\n3. Class distribution...")
    counts = df['Churn'].value_counts().sort_index()
    for cls, cnt in counts.items():
        print(f"   Class {int(cls)} ({CLASS_LABELS[int(cls)]}): {cnt:,} ({cnt/len(df)*100:.1f}%)")
    plot_class_distribution(df)

    # 4. Feature distributions
    print("\n4. Plotting feature distributions...")
    plot_feature_distributions(df, numerical_cols)

    # 5. Categorical distributions
    print("\n5. Plotting categorical distributions...")
    plot_categorical_distributions(df, categorical_cols)

    # 6. Correlation heatmap
    print("\n6. Plotting correlation heatmap...")
    plot_correlation_heatmap(df, numerical_cols, categorical_cols)

    # 7. Box plots
    print("\n7. Plotting box plots...")
    plot_boxplots(df, numerical_cols)

    # 8. Summary statistics
    print("\n8. Summary statistics (numerical features):")
    print(df[numerical_cols].describe().round(2).to_string())

    print("\n" + "=" * 60)
    print("EDA complete! All figures saved to figures/")
    print("=" * 60)


if __name__ == '__main__':
    run_eda()
