"""
preprocessing.py — Shared data loading and preprocessing module.

Used by all model scripts. Handles:
- Dataset download via kagglehub
- Cleaning (drop CustomerID, handle NaN)
- One-hot encoding of nominal categorical features
- StandardScaler normalisation
- Train/validation/test splitting
- Class weight computation
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def download_dataset():
    """Download dataset via kagglehub and return paths to train/test CSVs."""
    import kagglehub
    path = kagglehub.dataset_download("muhammadshahidazeem/customer-churn-dataset")
    train_path = os.path.join(path, "customer_churn_dataset-training-master.csv")
    test_path = os.path.join(path, "customer_churn_dataset-testing-master.csv")
    return train_path, test_path


def load_data(train_path, test_path):
    """Load train and test CSVs into DataFrames."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def preprocess_data(df, verbose=True):
    """
    Drop CustomerID (no predictive value) and rows with missing values.
    Returns cleaned DataFrame and a dict of cleaning statistics.
    """
    stats = {}
    stats['original_shape'] = df.shape
    stats['nan_per_col'] = df.isnull().sum().to_dict()
    stats['nan_rows'] = int(df.isnull().any(axis=1).sum())

    df = df.drop(columns=['CustomerID'])
    df = df.dropna()

    stats['cleaned_shape'] = df.shape

    if verbose:
        print(f"   Original shape: {stats['original_shape']}")
        print(f"   NaN rows dropped: {stats['nan_rows']}")
        print(f"   Cleaned shape:  {stats['cleaned_shape']}")

    return df, stats


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

NOMINAL_COLS = ['Gender', 'Subscription Type', 'Contract Length']


def encode_features(train, test, nominal_cols=None):
    """
    One-hot encode nominal categorical features using pd.get_dummies.
    Uses drop_first=True to avoid multicollinearity.

    Returns:
        train_encoded, test_encoded, feature_names (list of column names after encoding)
    """
    if nominal_cols is None:
        nominal_cols = NOMINAL_COLS

    # Ensure consistent categories across train/test
    train = train.copy()
    test = test.copy()

    train = pd.get_dummies(train, columns=nominal_cols, drop_first=True, dtype=int)
    test = pd.get_dummies(test, columns=nominal_cols, drop_first=True, dtype=int)

    # Align columns — in case test has missing/extra categories
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    feature_cols = [c for c in train.columns if c != 'Churn']
    return train, test, feature_cols


# ---------------------------------------------------------------------------
# Feature / Target Split
# ---------------------------------------------------------------------------

def split_features_target(df, target_col='Churn'):
    """Split DataFrame into feature matrix X and target vector y."""
    X = df.drop(columns=[target_col]).values.astype(np.float64)
    y = df[target_col].values.astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_features(X_train, X_test, X_val=None):
    """
    Fit StandardScaler on training data only, transform all sets.
    Returns scaled arrays and the fitted scaler.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    if X_val is not None:
        X_val_s = scaler.transform(X_val)
        return X_train_s, X_test_s, X_val_s, scaler
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Class Weights
# ---------------------------------------------------------------------------

def compute_class_weights(y):
    """
    Compute class weights inversely proportional to class frequency.
    Returns dict {class_label: weight}.
    """
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(cls): total / (len(classes) * cnt) for cls, cnt in zip(classes, counts)}
    return weights


# ---------------------------------------------------------------------------
# Full Pipeline
# ---------------------------------------------------------------------------

def full_pipeline(val_size=0.2, random_state=42, verbose=True):
    """
    Execute the complete preprocessing pipeline:
    1. Download/load data
    2. Clean
    3. One-hot encode
    4. Train/val split (stratified)
    5. Scale

    Returns:
        dict with keys:
            X_train, X_val, X_test, y_train, y_val, y_test,
            scaler, feature_names, class_weights, stats
    """
    if verbose:
        print("=" * 60)
        print("Data Preprocessing Pipeline")
        print("=" * 60)

    # 1. Load
    if verbose:
        print("\n1. Downloading / loading dataset...")
    train_path, test_path = download_dataset()
    train_raw, test_raw = load_data(train_path, test_path)
    if verbose:
        print(f"   Train raw: {train_raw.shape}, Test raw: {test_raw.shape}")

    # 2. Clean
    if verbose:
        print("\n2. Cleaning data...")
    train, train_stats = preprocess_data(train_raw, verbose=verbose)
    test, test_stats = preprocess_data(test_raw, verbose=verbose)

    # 3. Encode
    if verbose:
        print("\n3. One-hot encoding categorical features...")
        print(f"   Nominal columns: {NOMINAL_COLS}")
    train, test, feature_names = encode_features(train, test)
    if verbose:
        print(f"   Features after encoding: {len(feature_names)}")
        print(f"   Feature names: {feature_names}")

    # 4. Split features/target
    if verbose:
        print("\n4. Splitting features and target...")
    X_train_full, y_train_full = split_features_target(train)
    X_test, y_test = split_features_target(test)

    # 5. Train-validation split
    if verbose:
        print(f"\n5. Train-validation split ({1-val_size:.0%} / {val_size:.0%}, stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size, random_state=random_state, stratify=y_train_full
    )
    if verbose:
        print(f"   X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    # 6. Scale
    if verbose:
        print("\n6. Scaling features (StandardScaler, fit on train only)...")
    X_train, X_test, X_val, scaler = scale_features(X_train, X_test, X_val)

    # 7. Class weights
    class_weights = compute_class_weights(y_train)
    if verbose:
        classes, counts = np.unique(y_train, return_counts=True)
        print(f"\n7. Class distribution in training set:")
        for cls, cnt in zip(classes, counts):
            pct = cnt / len(y_train) * 100
            print(f"   Class {int(cls)}: {cnt:,} ({pct:.1f}%)")
        print(f"   Class weights: {class_weights}")

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'scaler': scaler, 'feature_names': feature_names,
        'class_weights': class_weights,
        'train_stats': train_stats, 'test_stats': test_stats,
    }
