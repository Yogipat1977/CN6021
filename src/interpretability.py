"""
interpretability.py — Feature Importance & Interpretability Analysis.

Methods:
1. Permutation Importance (global, across full test set)
2. SHAP (KernelExplainer) if available, else manual perturbation
3. Sensitivity Analysis (±1 std, averaged over test samples)
4. Partial Dependence Plots for top features

Group: Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from eda import (BG_DARK, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
                 BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE,
                 GRID_COLOR, save_fig)

# Apply theme
plt.rcParams.update({
    'figure.facecolor': BG_DARK, 'axes.facecolor': BG_PANEL,
    'axes.edgecolor': GRID_COLOR, 'axes.labelcolor': TEXT_PRIMARY,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'text.color': TEXT_PRIMARY,
    'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
    'legend.facecolor': BG_PANEL, 'legend.edgecolor': GRID_COLOR,
    'legend.fontsize': 9, 'legend.labelcolor': TEXT_PRIMARY,
    'grid.color': GRID_COLOR, 'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'figure.dpi': 150,
    'savefig.dpi': 200, 'savefig.facecolor': BG_DARK,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.3,
})


# ===================================================================
# Permutation Importance
# ===================================================================

def permutation_importance(model, X, y, feature_names, n_repeats=10, metric='f1'):
    """
    Compute permutation importance: for each feature, shuffle its values
    and measure the drop in performance. Repeat n_repeats times.

    Returns:
        importance_mean, importance_std (arrays of shape (n_features,))
    """
    print("\n   Computing permutation importance...")

    # Baseline score
    y_pred_base = model.predict(X)
    if metric == 'f1':
        baseline_score = f1_score(y, y_pred_base)
    else:
        baseline_score = np.mean(y == y_pred_base)

    n_features = X.shape[1]
    importances = np.zeros((n_repeats, n_features))

    for r in range(n_repeats):
        for j in range(n_features):
            X_permuted = X.copy()
            X_permuted[:, j] = np.random.permutation(X_permuted[:, j])
            y_pred_perm = model.predict(X_permuted)
            if metric == 'f1':
                perm_score = f1_score(y, y_pred_perm)
            else:
                perm_score = np.mean(y == y_pred_perm)
            importances[r, j] = baseline_score - perm_score

    imp_mean = importances.mean(axis=0)
    imp_std = importances.std(axis=0)

    # Print ranking
    sorted_idx = np.argsort(imp_mean)[::-1]
    print(f"   Baseline F1: {baseline_score:.4f}")
    print("   Feature ranking by permutation importance:")
    for rank, idx in enumerate(sorted_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        print(f"     {rank+1}. {name}: {imp_mean[idx]:.4f} ± {imp_std[idx]:.4f}")

    return imp_mean, imp_std, sorted_idx


def plot_permutation_importance(imp_mean, imp_std, feature_names, prefix='base'):
    """Bar chart of permutation importance with error bars."""
    sorted_idx = np.argsort(imp_mean)  # ascending for horizontal bar
    sorted_names = [feature_names[i] if i < len(feature_names) else f"F{i}"
                    for i in sorted_idx]
    sorted_mean = imp_mean[sorted_idx]
    sorted_std = imp_std[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.4)))
    bars = ax.barh(range(len(sorted_names)), sorted_mean,
                   xerr=sorted_std, color=BLUE_MID,
                   edgecolor=BG_DARK, linewidth=0.8, capsize=3,
                   error_kw={'color': BLUE_PALE, 'linewidth': 1.2})
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Mean F1-Score Decrease')
    ax.set_title('Permutation Feature Importance', pad=12)
    ax.axvline(x=0, color=TEXT_SECONDARY, linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_permutation_importance.png')


# ===================================================================
# Sensitivity Analysis (improved)
# ===================================================================

def sensitivity_analysis(model, X, feature_names, n_samples=500):
    """
    Perturb each feature by ±1 standard deviation (computed from X),
    averaged over n_samples data points. Much more robust than single
    sample + fixed perturbation.
    """
    print("\n   Computing sensitivity analysis...")

    # Use a subset of samples for efficiency
    if n_samples > X.shape[0]:
        n_samples = X.shape[0]
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_subset = X[indices]

    stds = np.std(X, axis=0)
    n_features = X.shape[1]
    sensitivities = np.zeros(n_features)

    base_preds = model.predict_proba(X_subset)

    for j in range(n_features):
        # Perturb by +1 std
        X_plus = X_subset.copy()
        X_plus[:, j] += stds[j]
        preds_plus = model.predict_proba(X_plus)

        # Perturb by -1 std
        X_minus = X_subset.copy()
        X_minus[:, j] -= stds[j]
        preds_minus = model.predict_proba(X_minus)

        # Average absolute change
        change = np.mean(np.abs(preds_plus - base_preds) + np.abs(preds_minus - base_preds)) / 2
        sensitivities[j] = change

    sorted_idx = np.argsort(sensitivities)[::-1]
    print("   Feature ranking by sensitivity:")
    for rank, idx in enumerate(sorted_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
        print(f"     {rank+1}. {name}: {sensitivities[idx]:.4f}")

    return sensitivities, sorted_idx


def plot_sensitivity_analysis(sensitivities, feature_names, prefix='base'):
    """Bar chart of sensitivity analysis."""
    sorted_idx = np.argsort(sensitivities)
    sorted_names = [feature_names[i] if i < len(feature_names) else f"F{i}"
                    for i in sorted_idx]
    sorted_vals = sensitivities[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.4)))
    ax.barh(range(len(sorted_names)), sorted_vals, color=BLUE_LIGHT,
            edgecolor=BG_DARK, linewidth=0.8)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Mean Prediction Change (±1σ Perturbation)')
    ax.set_title('Sensitivity Analysis', pad=12)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_sensitivity_analysis.png')


# ===================================================================
# SHAP Analysis
# ===================================================================

def run_shap_analysis(model, X_train, X_test, feature_names, prefix='base'):
    """
    Run SHAP KernelExplainer analysis. Falls back gracefully if SHAP
    is not installed.
    """
    try:
        import shap

        print("\n   Running SHAP analysis (this may take a few minutes)...")

        # Use a small background sample for KernelExplainer
        bg_size = min(100, X_train.shape[0])
        background = X_train[np.random.choice(X_train.shape[0], bg_size, replace=False)]

        # Create prediction function
        predict_fn = lambda x: model.predict_proba(x)

        explainer = shap.KernelExplainer(predict_fn, background)

        # Explain a subset of test data
        test_size = min(200, X_test.shape[0])
        X_explain = X_test[:test_size]
        shap_values = explainer.shap_values(X_explain)

        # Summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.4)))
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                          show=False, plot_type='dot', color_bar_label='Feature Value')
        # Adjust for dark theme
        ax = plt.gca()
        ax.set_facecolor(BG_PANEL)
        fig = plt.gcf()
        fig.set_facecolor(BG_DARK)
        for text in ax.get_yticklabels() + ax.get_xticklabels():
            text.set_color(TEXT_SECONDARY)
        ax.xaxis.label.set_color(TEXT_PRIMARY)
        ax.title.set_color(TEXT_PRIMARY) if ax.get_title() else None
        save_fig(fig, f'{prefix}_shap_summary.png')

        # Bar plot (mean |SHAP|)
        fig, ax = plt.subplots(figsize=(8, max(5, len(feature_names) * 0.4)))
        shap.summary_plot(shap_values, X_explain, feature_names=feature_names,
                          show=False, plot_type='bar')
        ax = plt.gca()
        ax.set_facecolor(BG_PANEL)
        fig = plt.gcf()
        fig.set_facecolor(BG_DARK)
        for text in ax.get_yticklabels() + ax.get_xticklabels():
            text.set_color(TEXT_SECONDARY)
        ax.xaxis.label.set_color(TEXT_PRIMARY)
        save_fig(fig, f'{prefix}_shap_bar.png')

        print("   SHAP analysis complete!")
        return shap_values

    except ImportError:
        print("   SHAP not installed — skipping SHAP analysis.")
        print("   Install with: pip install shap")
        return None
    except Exception as e:
        print(f"   SHAP analysis failed: {e}")
        print("   Falling back to manual methods only.")
        return None


# ===================================================================
# Partial Dependence Plots
# ===================================================================

def plot_partial_dependence(model, X, feature_names, top_n=3, prefix='base'):
    """
    Partial Dependence Plots for the top N most important features.
    Shows how average prediction changes as a single feature varies.
    """
    print(f"\n   Generating Partial Dependence Plots (top {top_n} features)...")

    # Use sensitivity to determine top features
    stds = np.std(X, axis=0)
    n_features = X.shape[1]
    sensitivities = np.zeros(n_features)
    base_preds = model.predict_proba(X[:500])

    for j in range(n_features):
        X_plus = X[:500].copy()
        X_plus[:, j] += stds[j]
        sensitivities[j] = np.mean(np.abs(model.predict_proba(X_plus) - base_preds))

    top_indices = np.argsort(sensitivities)[::-1][:top_n]

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
    if top_n == 1:
        axes = [axes]

    for ax, feat_idx in zip(axes, top_indices):
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature {feat_idx}"
        feat_vals = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 50)

        avg_preds = []
        for val in feat_vals:
            X_modified = X[:500].copy()
            X_modified[:, feat_idx] = val
            avg_preds.append(np.mean(model.predict_proba(X_modified)))

        ax.plot(feat_vals, avg_preds, color=BLUE_LIGHT, linewidth=2)
        ax.fill_between(feat_vals, avg_preds, alpha=0.1, color=BLUE_LIGHT)
        ax.set_xlabel(feat_name)
        ax.set_ylabel('Avg. Churn Probability')
        ax.set_title(f'PDP: {feat_name}', fontsize=11)
        ax.grid(alpha=0.3)

    fig.suptitle('Partial Dependence Plots', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_partial_dependence.png')


# ===================================================================
# Main runner
# ===================================================================

def run_interpretability(model, X_train, X_test, y_test, feature_names, prefix='base'):
    """Run the complete interpretability analysis."""
    print("\n" + "=" * 60)
    print("Interpretability Analysis")
    print("=" * 60)

    # 1. Permutation importance
    imp_mean, imp_std, sorted_idx = permutation_importance(
        model, X_test, y_test, feature_names, n_repeats=10
    )
    plot_permutation_importance(imp_mean, imp_std, feature_names, prefix)

    # 2. Sensitivity analysis
    sensitivities, _ = sensitivity_analysis(model, X_test, feature_names, n_samples=500)
    plot_sensitivity_analysis(sensitivities, feature_names, prefix)

    # 3. SHAP
    shap_values = run_shap_analysis(model, X_train, X_test, feature_names, prefix)

    # 4. Partial dependence plots
    plot_partial_dependence(model, X_test, feature_names, top_n=3, prefix=prefix)

    print("\n" + "=" * 60)
    print("Interpretability Analysis Complete!")
    print("=" * 60)

    return {
        'permutation_mean': imp_mean,
        'permutation_std': imp_std,
        'sensitivities': sensitivities,
        'shap_values': shap_values,
    }


if __name__ == '__main__':
    # Standalone run: train base model first, then interpret
    from shallow_nn import main as train_main
    model, metrics, history, data = train_main()

    results = run_interpretability(
        model, data['X_train'], data['X_test'], data['y_test'],
        data['feature_names'], prefix='base'
    )
