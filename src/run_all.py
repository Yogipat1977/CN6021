"""
run_all.py — Single entry-point to run the complete pipeline.

Steps:
1. Run EDA → generate exploration figures
2. Train base model → evaluate → generate plots
3. Run interpretability analysis → generate importance plots

Tuning is done separately via shallow_nn_tuned.py AFTER reviewing
base model results.

Group: Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)
"""

import os
import sys

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("\n" + "█" * 60)
    print("  CN6021 — Customer Churn Prediction Pipeline")
    print("  Group: Hard Joshi, Jayrup Nakawala, Yogi Patel")
    print("█" * 60)

    # ---- Step 1: EDA ----
    print("\n\n" + "▓" * 60)
    print("  STEP 1 / 3 : Exploratory Data Analysis")
    print("▓" * 60)
    from eda import run_eda
    run_eda()

    # ---- Step 2: Train base model ----
    print("\n\n" + "▓" * 60)
    print("  STEP 2 / 3 : Train & Evaluate Base Model")
    print("▓" * 60)
    from shallow_nn import main as train_main
    model, metrics, history, data = train_main()

    # ---- Step 3: Interpretability ----
    print("\n\n" + "▓" * 60)
    print("  STEP 3 / 3 : Interpretability Analysis")
    print("▓" * 60)
    from interpretability import run_interpretability
    interp_results = run_interpretability(
        model, data['X_train'], data['X_test'], data['y_test'],
        data['feature_names'], prefix='base'
    )

    # ---- Summary ----
    print("\n\n" + "█" * 60)
    print("  PIPELINE COMPLETE — Summary")
    print("█" * 60)
    print(f"\n  Base Model Results:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print(f"    AUC-ROC:   {metrics['auc']:.4f}")
    print(f"\n  Figures saved to: figures/")
    print(f"\n  Next step: Review results and run shallow_nn_tuned.py")
    print("█" * 60 + "\n")

    return model, metrics, history, data, interp_results


if __name__ == '__main__':
    main()
