"""
shallow_nn.py — Shallow Neural Network for Customer Churn Prediction.

Pure NumPy implementation with manual forward pass, backpropagation, and
Adam optimiser. Meets CN6021 requirement: "Develop using NumPy (or similar)".

Architecture:  input → hidden (Tanh) → output (Sigmoid)
Loss:          Binary Cross-Entropy with class weighting
Optimiser:     Adam (implemented from scratch)

Group: Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import seaborn as sns

# Import shared theme constants
sys.path.insert(0, os.path.dirname(__file__))
from eda import (BG_DARK, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
                 BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE, BLUE_CMAP,
                 GRID_COLOR, save_fig)

# Apply same theme
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


FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')

np.random.seed(42)


# ===================================================================
# Activation functions and their derivatives
# ===================================================================

def tanh(z):
    """Tanh activation: maps inputs to (-1, 1)."""
    return np.tanh(z)

def tanh_derivative(a):
    """Derivative of tanh given the activation output a = tanh(z)."""
    return 1.0 - a ** 2

def sigmoid(z):
    """Sigmoid activation: maps inputs to (0, 1). Numerically stable."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


# ===================================================================
# Loss function
# ===================================================================

def binary_cross_entropy(y_true, y_pred, sample_weights=None):
    """
    Weighted binary cross-entropy loss.
    L = -1/N * Σ w_i [y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i)]
    """
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss_per_sample = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if sample_weights is not None:
        loss_per_sample *= sample_weights
    return np.mean(loss_per_sample)


# ===================================================================
# Shallow Neural Network class
# ===================================================================

class ShallowNeuralNetwork:
    """
    Single hidden-layer neural network implemented in pure NumPy.

    Architecture:
        Input (n_features) → Hidden (hidden_size, Tanh) → Output (1, Sigmoid)

    Training:
        - Adam optimiser (manual implementation)
        - Weighted BCE loss for class imbalance
        - Early stopping with patience
        - Mini-batch gradient descent
    """

    def __init__(self, input_size, hidden_size=32, learning_rate=0.01,
                 weight_decay=0.001):
        """
        Initialise network parameters with Xavier/Glorot initialisation.

        Xavier init:  W ~ U(-sqrt(6 / (fan_in + fan_out)), sqrt(6 / (fan_in + fan_out)))
        This prevents vanishing/exploding gradients in combination with Tanh.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.weight_decay = weight_decay

        # Xavier initialisation
        limit1 = np.sqrt(6.0 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))

        limit2 = np.sqrt(6.0 / (hidden_size + 1))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, 1))
        self.b2 = np.zeros((1, 1))

        # Adam optimiser state
        self._init_adam()

    def _init_adam(self):
        """Initialise Adam optimiser moment estimates."""
        self.t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        # First moment (mean of gradients)
        self.m = {
            'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2),
        }
        # Second moment (mean of squared gradients)
        self.v = {
            'W1': np.zeros_like(self.W1), 'b1': np.zeros_like(self.b1),
            'W2': np.zeros_like(self.W2), 'b2': np.zeros_like(self.b2),
        }

    def forward(self, X):
        """
        Forward pass.
        Z1 = X @ W1 + b1     (linear transform)
        A1 = tanh(Z1)         (hidden activation)
        Z2 = A1 @ W2 + b2     (linear transform)
        A2 = sigmoid(Z2)      (output probability)
        """
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, y_true, sample_weights=None):
        """
        Backpropagation — compute gradients for all parameters.

        Output layer gradient:
            dL/dZ2 = (A2 - y) * w_i  (derivative of BCE + sigmoid combined)

        Hidden layer gradient:
            dL/dZ1 = (dL/dZ2 @ W2^T) ⊙ tanh'(A1)

        Parameter gradients:
            dW2 = A1^T @ dL/dZ2 / N
            db2 = mean(dL/dZ2)
            dW1 = X^T @ dL/dZ1 / N
            db1 = mean(dL/dZ1)
        """
        m = y_true.shape[0]
        y_true = y_true.reshape(-1, 1)

        # Output layer
        dZ2 = self.A2 - y_true  # shape: (m, 1)
        if sample_weights is not None:
            dZ2 *= sample_weights.reshape(-1, 1)

        dW2 = (self.A1.T @ dZ2) / m
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        # Hidden layer
        dA1 = dZ2 @ self.W2.T                        # shape: (m, hidden_size)
        dZ1 = dA1 * tanh_derivative(self.A1)          # element-wise

        dW1 = (self.X.T @ dZ1) / m
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # L2 regularisation gradients
        dW2 += self.weight_decay * self.W2
        dW1 += self.weight_decay * self.W1

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def _adam_update(self, grads):
        """
        Adam optimiser update step.
        Kingma & Ba (2015): Adaptive learning rates with momentum and RMSProp.

        m_t = β1·m_{t-1} + (1-β1)·g_t          (first moment)
        v_t = β2·v_{t-1} + (1-β2)·g_t²         (second moment)
        m̂_t = m_t / (1 - β1^t)                  (bias correction)
        v̂_t = v_t / (1 - β2^t)                  (bias correction)
        θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε)  (parameter update)
        """
        self.t += 1
        for param_name in ['W1', 'b1', 'W2', 'b2']:
            g = grads[param_name]

            # Update moments
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * g
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * g**2

            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            # Update parameter
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            param = getattr(self, param_name)
            setattr(self, param_name, param - update)

    def predict_proba(self, X):
        """Return probability predictions (forward pass without caching)."""
        Z1 = X @ self.W1 + self.b1
        A1 = tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)
        return A2.flatten()

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self):
        """Return a copy of all weights (for saving best model)."""
        return {
            'W1': self.W1.copy(), 'b1': self.b1.copy(),
            'W2': self.W2.copy(), 'b2': self.b2.copy(),
        }

    def set_weights(self, weights):
        """Restore weights from a saved copy."""
        self.W1 = weights['W1'].copy()
        self.b1 = weights['b1'].copy()
        self.W2 = weights['W2'].copy()
        self.b2 = weights['b2'].copy()


# ===================================================================
# Metrics
# ===================================================================

def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics manually."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    auc_roc = roc_auc_score(y_true, y_proba) if y_proba is not None else 0.0

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1, 'auc': auc_roc,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


# ===================================================================
# Training loop
# ===================================================================

def train_model(model, X_train, y_train, X_val, y_val,
                class_weights, epochs=100, batch_size=2048, patience=15):
    """
    Train the shallow NN with mini-batch gradient descent, weighted loss,
    and early stopping.

    Args:
        model: ShallowNeuralNetwork instance
        class_weights: dict {0: w0, 1: w1}
        patience: number of epochs without val loss improvement before stopping

    Returns:
        history dict with train_loss, val_loss, val_f1 per epoch
    """
    # Precompute per-sample weights for training set
    sample_weights_train = np.array([class_weights[int(y)] for y in y_train])
    sample_weights_val = np.array([class_weights[int(y)] for y in y_val])

    n_samples = X_train.shape[0]
    n_batches = max(1, n_samples // batch_size)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    best_val_loss = np.inf
    best_weights = None
    patience_counter = 0

    print(f"\n   Training: {epochs} max epochs, batch_size={batch_size}, patience={patience}")
    print(f"   Architecture: {model.input_size} → {model.hidden_size} → 1")
    print(f"   LR={model.lr}, L2={model.weight_decay}")

    start_time = time.time()

    for epoch in range(epochs):
        # Shuffle training data each epoch
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        w_shuffled = sample_weights_train[perm]

        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)

            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            w_batch = w_shuffled[start:end]

            # Forward
            y_pred = model.forward(X_batch)

            # Loss
            batch_loss = binary_cross_entropy(y_batch, y_pred.flatten(), w_batch)
            epoch_loss += batch_loss

            # Backward
            grads = model.backward(y_batch, w_batch)

            # Update
            model._adam_update(grads)

        # Epoch-level metrics
        train_loss = epoch_loss / n_batches

        # Validation
        val_proba = model.predict_proba(X_val)
        val_loss = binary_cross_entropy(y_val, val_proba, sample_weights_val)
        val_pred = (val_proba >= 0.5).astype(int)
        val_metrics = compute_metrics(y_val, val_pred, val_proba)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter >= patience:
            elapsed = time.time() - start_time
            print(f"   Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f} | "
                  f"Time: {elapsed:.1f}s")

        if patience_counter >= patience:
            print(f"   → Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)
        print(f"   → Restored best weights (val_loss={best_val_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"   → Training completed in {elapsed:.1f}s ({epoch+1} epochs)")

    return history


# ===================================================================
# Plotting functions
# ===================================================================

def plot_training_curves(history, prefix='base'):
    """Plot training and validation loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax1.plot(epochs, history['train_loss'], color=BLUE_MID, linewidth=1.8,
             label='Training Loss', alpha=0.9)
    ax1.plot(epochs, history['val_loss'], color=BLUE_LIGHT, linewidth=1.8,
             label='Validation Loss', alpha=0.9)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Validation F1 + AUC
    ax2.plot(epochs, history['val_f1'], color=BLUE_LIGHT, linewidth=1.8,
             label='Val F1-Score', alpha=0.9)
    ax2.plot(epochs, history['val_auc'], color=BLUE_PALE, linewidth=1.8,
             label='Val AUC-ROC', linestyle='--', alpha=0.9)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f'Training History ({prefix.replace("_", " ").title()})',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_training_curves.png')


def plot_roc_curve(y_true, y_proba, prefix='base'):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=BLUE_LIGHT, linewidth=2, label=f'ROC (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=1, linestyle='--',
            alpha=0.5, label='Random Baseline')
    ax.fill_between(fpr, tpr, alpha=0.1, color=BLUE_LIGHT)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_fig(fig, f'{prefix}_roc_curve.png')


def plot_pr_curve(y_true, y_proba, prefix='base'):
    """Plot Precision-Recall curve."""
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec, prec)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color=BLUE_LIGHT, linewidth=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color=TEXT_SECONDARY, linewidth=1, linestyle='--',
               alpha=0.5, label=f'Baseline ({baseline:.3f})')
    ax.fill_between(rec, prec, alpha=0.1, color=BLUE_LIGHT)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_fig(fig, f'{prefix}_pr_curve.png')


def plot_confusion_matrix(metrics, prefix='base'):
    """Plot confusion matrix heatmap."""
    cm = np.array([
        [metrics['tn'], metrics['fp']],
        [metrics['fn'], metrics['tp']]
    ])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap=BLUE_CMAP,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'],
                ax=ax, linewidths=2, linecolor=BG_DARK,
                annot_kws={'size': 14, 'fontweight': 'bold', 'color': TEXT_PRIMARY},
                cbar_kws={'shrink': 0.8})
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
    ax.tick_params(colors=TEXT_SECONDARY)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_confusion_matrix.png')


# ===================================================================
# Main
# ===================================================================

def main():
    """Run the base shallow neural network pipeline."""
    print("=" * 60)
    print("Customer Churn Prediction — Shallow Neural Network (NumPy)")
    print("=" * 60)

    # 1. Load preprocessed data
    from preprocessing import full_pipeline
    data = full_pipeline(val_size=0.2, random_state=42)

    X_train = data['X_train']
    X_val   = data['X_val']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_val   = data['y_val']
    y_test  = data['y_test']
    class_weights = data['class_weights']
    feature_names = data['feature_names']

    print(f"\n   Feature count: {X_train.shape[1]}")
    print(f"   Feature names: {feature_names}")

    # 2. Create and train model
    print("\n" + "=" * 60)
    print("Training Base Model")
    print("=" * 60)

    model = ShallowNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=32,
        learning_rate=0.01,
        weight_decay=0.001
    )

    history = train_model(
        model, X_train, y_train, X_val, y_val,
        class_weights=class_weights,
        epochs=100,
        batch_size=2048,
        patience=15
    )

    # 3. Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=0.5)
    metrics = compute_metrics(y_test, y_pred, y_proba)

    print(f"\n   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {metrics['auc']:.4f}")
    print(f"   TP: {metrics['tp']:,}  TN: {metrics['tn']:,}  "
          f"FP: {metrics['fp']:,}  FN: {metrics['fn']:,}")

    # 4. Generate plots
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)

    plot_training_curves(history, prefix='base')
    plot_roc_curve(y_test, y_proba, prefix='base')
    plot_pr_curve(y_test, y_proba, prefix='base')
    plot_confusion_matrix(metrics, prefix='base')

    print("\n" + "=" * 60)
    print("Base Model Complete!")
    print("=" * 60)

    return model, metrics, history, data


if __name__ == '__main__':
    model, metrics, history, data = main()