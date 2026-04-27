"""
shallow_nn_tuned.py — Tuned Shallow Neural Network for Customer Churn.

Tuning strategy (informed by base model diagnosis):
──────────────────────────────────────────────────────
Base model showed:
  - Excellent validation performance (F1=0.98, AUC=0.997)
  - Poor test generalisation (F1=0.66, AUC=0.75)
  - Over-prediction of class 1 (94.7% predicted positive)

Root cause: Train-test distribution shift + overfitting to training patterns.

Tuning focus:
  1. Stronger regularisation (higher weight_decay) to combat overfitting
  2. Smaller hidden sizes to reduce model capacity
  3. Lower learning rates for more stable convergence
  4. Threshold optimisation to fix the decision boundary
  5. Stratified K-fold CV for reliable model selection

Group: Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import full_pipeline, compute_class_weights
from shallow_nn import (ShallowNeuralNetwork, train_model, compute_metrics,
                         plot_training_curves, plot_roc_curve, plot_pr_curve,
                         plot_confusion_matrix, binary_cross_entropy)
from eda import (BG_DARK, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
                 BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE,
                 GRID_COLOR, BLUE_CMAP, save_fig)

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

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
np.random.seed(42)


# ===================================================================
# Stratified K-Fold Cross-Validation
# ===================================================================

def cross_validate(X, y, hidden_size, lr, weight_decay, n_folds=5,
                   epochs=150, batch_size=2048, patience=15, verbose=True):
    """
    Perform stratified K-fold cross-validation.

    Why Stratified K-Fold?
    ──────────────────────
    Standard K-fold randomly partitions data, which can create folds with
    skewed class ratios—especially problematic for imbalanced datasets.
    Stratified K-Fold guarantees that each fold preserves the original
    class distribution (43.3% retained / 56.7% churned), ensuring:

    1. Each fold is a representative sample of the full dataset
    2. Evaluation metrics are unbiased and comparable across folds
    3. The model sees consistent class proportions during training

    This is particularly important here because our class ratio (43:57)
    would cause significant variance across folds without stratification.

    Returns:
        dict with mean ± std for each metric across folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        class_weights = compute_class_weights(y_fold_train)

        model = ShallowNeuralNetwork(
            input_size=X.shape[1],
            hidden_size=hidden_size,
            learning_rate=lr,
            weight_decay=weight_decay
        )

        # Suppress per-epoch output during CV
        import io
        from contextlib import redirect_stdout
        f = io.StringIO()
        with redirect_stdout(f):
            train_model(model, X_fold_train, y_fold_train,
                       X_fold_val, y_fold_val, class_weights,
                       epochs=epochs, batch_size=batch_size, patience=patience)

        y_proba = model.predict_proba(X_fold_val)
        y_pred = model.predict(X_fold_val)
        metrics = compute_metrics(y_fold_val, y_pred, y_proba)
        fold_metrics.append(metrics)

        if verbose:
            print(f"      Fold {fold+1}/{n_folds}: "
                  f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, "
                  f"Acc={metrics['accuracy']:.4f}")

    # Aggregate
    result = {}
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        vals = [m[key] for m in fold_metrics]
        result[f'{key}_mean'] = np.mean(vals)
        result[f'{key}_std'] = np.std(vals)

    if verbose:
        print(f"      Mean F1: {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
        print(f"      Mean AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")

    return result


# ===================================================================
# Hyperparameter Grid Search
# ===================================================================

def grid_search(X_train, y_train, param_grid, n_folds=5):
    """
    Exhaustive grid search over hyperparameter combinations with
    stratified cross-validation for reliable model selection.
    """
    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(product(*values))

    print(f"\n   Grid search: {len(all_combos)} combinations × {n_folds} folds")
    print(f"   Estimated time: ~{len(all_combos) * n_folds * 13 / 60:.0f} minutes")

    results = []
    best_f1 = 0
    best_config = None
    start = time.time()

    for i, combo in enumerate(all_combos):
        config = dict(zip(keys, combo))
        print(f"\n   [{i+1}/{len(all_combos)}] hidden={config['hidden_size']}, "
              f"lr={config['learning_rate']}, wd={config['weight_decay']}")

        cv_result = cross_validate(
            X_train, y_train,
            hidden_size=config['hidden_size'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            n_folds=n_folds,
            epochs=150,
            patience=15
        )
        cv_result['config'] = config
        results.append(cv_result)

        if cv_result['f1_mean'] > best_f1:
            best_f1 = cv_result['f1_mean']
            best_config = config

        elapsed = time.time() - start
        remaining = elapsed / (i + 1) * (len(all_combos) - i - 1)
        print(f"      Elapsed: {elapsed/60:.1f}min | Remaining: ~{remaining/60:.1f}min")

    print(f"\n   ✓ Best config: {best_config} (CV F1={best_f1:.4f})")
    print(f"   ✓ Total search time: {(time.time()-start)/60:.1f} minutes")
    return results, best_config


def plot_grid_search_results(results, prefix='tuned'):
    """Heatmap of grid search results (F1 by hidden_size × learning_rate)."""
    import pandas as pd
    import seaborn as sns

    records = []
    for r in results:
        cfg = r['config']
        records.append({
            'hidden_size': cfg['hidden_size'],
            'learning_rate': cfg['learning_rate'],
            'weight_decay': cfg['weight_decay'],
            'f1_mean': r['f1_mean'],
            'auc_mean': r['auc_mean'],
        })
    df = pd.DataFrame(records)

    # Pivot for heatmap (aggregate over weight_decay by taking best)
    pivot = df.groupby(['hidden_size', 'learning_rate'])['f1_mean'].max().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap=BLUE_CMAP, ax=ax,
                linewidths=2, linecolor=BG_DARK,
                annot_kws={'size': 11, 'fontweight': 'bold', 'color': TEXT_PRIMARY},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Grid Search: Best F1 by Hidden Size × Learning Rate',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Hidden Size')
    ax.set_xlabel('Learning Rate')
    ax.tick_params(colors=TEXT_SECONDARY)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_grid_search_heatmap.png')

    # Also plot weight_decay effect
    pivot_wd = df.groupby(['hidden_size', 'weight_decay'])['f1_mean'].max().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_wd, annot=True, fmt='.4f', cmap=BLUE_CMAP, ax=ax,
                linewidths=2, linecolor=BG_DARK,
                annot_kws={'size': 11, 'fontweight': 'bold', 'color': TEXT_PRIMARY},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Grid Search: Best F1 by Hidden Size × Weight Decay',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Hidden Size')
    ax.set_xlabel('Weight Decay (L2)')
    ax.tick_params(colors=TEXT_SECONDARY)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_grid_search_wd_heatmap.png')


# ===================================================================
# Threshold Optimisation
# ===================================================================

def optimise_threshold(model, X_val, y_val, prefix='tuned'):
    """
    Sweep prediction thresholds to find the one that maximises F1-score.
    This addresses the base model's issue of predicting almost everything
    as class 1 by finding a better decision boundary.
    """
    print("\n   Optimising decision threshold...")
    y_proba = model.predict_proba(X_val)
    thresholds = np.arange(0.20, 0.80, 0.01)
    best_f1, best_thresh = 0, 0.5

    thresh_results = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        m = compute_metrics(y_val, y_pred, y_proba)
        thresh_results.append((thresh, m['f1'], m['precision'], m['recall'], m['accuracy']))
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_thresh = thresh

    print(f"   Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")

    # Plot
    thresh_arr = np.array(thresh_results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresh_arr[:, 0], thresh_arr[:, 1], color=BLUE_LIGHT, linewidth=2.2, label='F1-Score')
    ax.plot(thresh_arr[:, 0], thresh_arr[:, 2], color=BLUE_MID, linewidth=1.5,
            label='Precision', linestyle='--')
    ax.plot(thresh_arr[:, 0], thresh_arr[:, 3], color=BLUE_PALE, linewidth=1.5,
            label='Recall', linestyle=':')
    ax.plot(thresh_arr[:, 0], thresh_arr[:, 4], color=TEXT_SECONDARY, linewidth=1.2,
            label='Accuracy', linestyle='-', alpha=0.6)
    ax.axvline(x=best_thresh, color='#FF6B6B', linewidth=1.5, linestyle='-.',
               alpha=0.8, label=f'Optimal ({best_thresh:.2f})')
    ax.axvline(x=0.5, color=TEXT_SECONDARY, linewidth=1, linestyle=':',
               alpha=0.4, label='Default (0.50)')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Optimisation — Precision-Recall Trade-off')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_threshold_optimisation.png')

    return best_thresh


# ===================================================================
# Comparison Plots
# ===================================================================

def plot_comparison(base_metrics, tuned_metrics, prefix='comparison'):
    """Side-by-side bar chart comparing base vs tuned model metrics."""
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']

    base_vals = [base_metrics[k] for k in metrics_keys]
    tuned_vals = [tuned_metrics[k] for k in metrics_keys]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, base_vals, width, label='Base Model',
                   color=BLUE_DARK, edgecolor=BG_DARK, linewidth=1)
    bars2 = ax.bar(x + width/2, tuned_vals, width, label='Tuned Model',
                   color=BLUE_LIGHT, edgecolor=BG_DARK, linewidth=1)

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9,
                color=TEXT_PRIMARY, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Base vs Tuned Model — Test Set Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.12)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_model_comparison.png')


def plot_roc_comparison(y_test, base_proba, tuned_proba, prefix='comparison'):
    """Overlaid ROC curves for base and tuned models."""
    fpr_b, tpr_b, _ = roc_curve(y_test, base_proba)
    fpr_t, tpr_t, _ = roc_curve(y_test, tuned_proba)
    auc_b = roc_auc_score(y_test, base_proba)
    auc_t = roc_auc_score(y_test, tuned_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_b, tpr_b, color=BLUE_DARK, linewidth=2, label=f'Base (AUC={auc_b:.4f})')
    ax.plot(fpr_t, tpr_t, color=BLUE_LIGHT, linewidth=2, label=f'Tuned (AUC={auc_t:.4f})')
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_roc_comparison.png')


# ===================================================================
# Main
# ===================================================================

def main(base_metrics=None, base_proba=None):
    """
    Run the full tuning pipeline:
    1. Load preprocessed data
    2. Grid search with stratified 5-fold CV
    3. Train best model on training set
    4. Optimise decision threshold on validation set
    5. Final evaluation on held-out test set
    6. Generate all comparison plots
    7. Run interpretability on tuned model
    """
    print("=" * 60)
    print("Customer Churn Prediction — Tuned Shallow NN")
    print("=" * 60)

    # 1. Load data (same pipeline as base model for fair comparison)
    data = full_pipeline(val_size=0.2, random_state=42)
    X_train = data['X_train']
    X_val   = data['X_val']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_val   = data['y_val']
    y_test  = data['y_test']
    class_weights = data['class_weights']
    feature_names = data['feature_names']

    # 2. Grid search with cross-validation
    # Grid is targeted based on base model analysis:
    # - Smaller hidden sizes (16, 32) to reduce overfitting
    # - Range of LRs including lower values for stability
    # - Higher weight_decay values for stronger regularisation
    print("\n" + "=" * 60)
    print("Hyperparameter Grid Search (Stratified 5-Fold CV)")
    print("=" * 60)

    param_grid = {
        'hidden_size':   [16, 32, 64],
        'learning_rate': [0.001, 0.005, 0.01],
        'weight_decay':  [0.001, 0.01, 0.05],
    }

    # Combine train + val for CV (test is held out for final eval only)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    cv_results, best_config = grid_search(X_trainval, y_trainval, param_grid, n_folds=5)
    plot_grid_search_results(cv_results, prefix='tuned')

    # 3. Train final model with best config
    print("\n" + "=" * 60)
    print(f"Training Final Model: {best_config}")
    print("=" * 60)

    best_model = ShallowNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=best_config['hidden_size'],
        learning_rate=best_config['learning_rate'],
        weight_decay=best_config['weight_decay']
    )

    history = train_model(
        best_model, X_train, y_train, X_val, y_val,
        class_weights=class_weights,
        epochs=200, batch_size=2048, patience=20
    )

    # 4. Threshold optimisation
    print("\n" + "=" * 60)
    print("Threshold Optimisation")
    print("=" * 60)
    best_threshold = optimise_threshold(best_model, X_val, y_val, prefix='tuned')

    # 5. Final evaluation on test set
    print("\n" + "=" * 60)
    print(f"Test Evaluation (threshold={best_threshold:.2f})")
    print("=" * 60)

    y_proba = best_model.predict_proba(X_test)
    y_pred = (y_proba >= best_threshold).astype(int)
    tuned_metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\n   Accuracy:  {tuned_metrics['accuracy']:.4f}")
    print(f"   Precision: {tuned_metrics['precision']:.4f}")
    print(f"   Recall:    {tuned_metrics['recall']:.4f}")
    print(f"   F1-Score:  {tuned_metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {tuned_metrics['auc']:.4f}")
    print(f"   TP: {tuned_metrics['tp']:,}  TN: {tuned_metrics['tn']:,}  "
          f"FP: {tuned_metrics['fp']:,}  FN: {tuned_metrics['fn']:,}")

    # Also show default threshold for comparison
    y_pred_default = (y_proba >= 0.5).astype(int)
    default_metrics = compute_metrics(y_test, y_pred_default, y_proba)
    print(f"\n   [For comparison — default threshold=0.50]")
    print(f"   Accuracy:  {default_metrics['accuracy']:.4f}  |  "
          f"F1: {default_metrics['f1']:.4f}  |  "
          f"Prec: {default_metrics['precision']:.4f}  |  "
          f"Rec: {default_metrics['recall']:.4f}")

    # 6. Generate plots
    print("\n   Generating plots...")
    plot_training_curves(history, prefix='tuned')
    plot_roc_curve(y_test, y_proba, prefix='tuned')
    plot_pr_curve(y_test, y_proba, prefix='tuned')
    plot_confusion_matrix(tuned_metrics, prefix='tuned')

    # 7. Comparison with base model
    if base_metrics is not None:
        print("\n   Generating comparison plots...")
        plot_comparison(base_metrics, tuned_metrics, prefix='comparison')
        if base_proba is not None:
            plot_roc_comparison(y_test, base_proba, y_proba, prefix='comparison')

    # 8. Interpretability on tuned model
    from interpretability import run_interpretability
    run_interpretability(best_model, X_train, X_test, y_test, feature_names, prefix='tuned')

    print("\n" + "=" * 60)
    print("Tuned Model Complete!")
    print(f"  Best config:      {best_config}")
    print(f"  Optimal threshold: {best_threshold:.2f}")
    print(f"  Test F1:          {tuned_metrics['f1']:.4f}")
    print(f"  Test AUC:         {tuned_metrics['auc']:.4f}")
    print("=" * 60)

    return best_model, tuned_metrics, best_config, best_threshold, y_proba


if __name__ == '__main__':
    # Run base model first for comparison
    print("Step 1: Running base model for comparison...")
    from shallow_nn import main as base_main
    base_model, base_metrics, base_history, data = base_main()
    base_proba = base_model.predict_proba(data['X_test'])

    # Run tuned model
    print("\n\nStep 2: Running tuned model...")
    tuned_model, tuned_metrics, best_config, best_threshold, tuned_proba = main(
        base_metrics=base_metrics, base_proba=base_proba
    )