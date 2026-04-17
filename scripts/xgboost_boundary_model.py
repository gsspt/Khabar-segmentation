#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost model for isnad boundary prediction.

Train a regression model to predict the correct isnad end position
given features extracted from the detected isnad start.
"""

import json
import csv
from pathlib import Path
from typing import Tuple, Dict
import numpy as np

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import pickle
except ImportError:
    print("[ERROR] Missing required libraries. Install with:")
    print("  pip install xgboost scikit-learn numpy")
    exit(1)


def load_features_from_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load features and target from CSV.

    Returns:
        (X_features, y_target, feature_names)
    """
    features_list = []
    targets = []
    feature_names = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        feature_names = [col for col in reader.fieldnames if col != 'target_offset']

        for row in reader:
            # Extract features
            feature_values = [float(row.get(col, 0)) for col in feature_names]
            features_list.append(feature_values)

            # Extract target
            target = float(row.get('target_offset', 0))
            targets.append(target)

    X = np.array(features_list)
    y = np.array(targets)

    return X, y, feature_names


def train_xgboost_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None) -> xgb.XGBRegressor:
    """
    Train XGBoost regression model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: (Optional) Validation features
        y_val: (Optional) Validation targets

    Returns:
        Trained XGBoost model
    """
    print("[*] Training XGBoost model...")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val) if X_val is not None else None

    # Parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }

    # Train with early stopping
    evals = [(dtrain, 'train')]
    if dval is not None:
        evals.append((dval, 'val'))

    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    return model, evals_result


def evaluate_model(model: xgb.Booster, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """
    Evaluate model on test set.

    Returns:
        Dict of metrics
    """
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Custom metric: % within tolerance
    tolerances = [20, 50, 100]
    within_tolerance = {}
    for tol in tolerances:
        within = np.sum(np.abs(y_test - y_pred) <= tol) / len(y_test) * 100
        within_tolerance[f'within_{tol}'] = within

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test,
        **within_tolerance
    }


def main():
    """Train and evaluate XGBoost boundary model."""
    print("=" * 80)
    print("XGBOOST BOUNDARY PREDICTION MODEL")
    print("=" * 80 + "\n")

    # Check if features exist
    features_path = Path("../data/processed/xgboost_training_features.csv")
    if not features_path.exists():
        print("[ERROR] Features file not found. Run xgboost_feature_extraction.py first.")
        exit(1)

    # Load features
    print("[1] Loading features...")
    X, y, feature_names = load_features_from_csv(features_path)
    print(f"    Samples: {len(X)}")
    print(f"    Features: {len(feature_names)}")
    print(f"    Target range: {y.min():.0f} - {y.max():.0f} chars\n")

    # Split data
    print("[2] Splitting data (70% train, 15% val, 15% test)...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"    Train: {len(X_train)} samples")
    print(f"    Val:   {len(X_val)} samples")
    print(f"    Test:  {len(X_test)} samples\n")

    # Train model
    print("[3] Training model...")
    model, evals_result = train_xgboost_model(X_train, y_train, X_val, y_val)
    print("    [OK] Model trained\n")

    # Evaluate
    print("[4] Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"    MAE (Mean Absolute Error):  {metrics['mae']:.1f} chars")
    print(f"    RMSE (Root Mean Sq Error):  {metrics['rmse']:.1f} chars")
    print(f"    R-squared:                  {metrics['r2']:.3f}")
    print(f"    Within 20 chars:            {metrics['within_20']:.1f}%")
    print(f"    Within 50 chars:            {metrics['within_50']:.1f}%")
    print(f"    Within 100 chars:           {metrics['within_100']:.1f}%\n")

    # Feature importance
    print("[5] Feature importance (top 10):")
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    for i, (feat, score) in enumerate(sorted_importance[:10]):
        feat_name = feature_names[int(feat.replace('f', ''))] if feat.startswith('f') else feat
        print(f"    {i+1}. {feat_name}: {score:.1f}")

    # Save model
    print("\n[6] Saving model...")
    model_path = Path("../results/xgboost_boundary_model.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))

    # Save feature names mapping
    feature_map_path = Path("../results/xgboost_feature_names.json")
    with open(feature_map_path, 'w', encoding='utf-8') as f:
        json.dump({
            'feature_names': feature_names,
            'feature_count': len(feature_names),
            'metrics': {
                'mae': float(metrics['mae']),
                'rmse': float(metrics['rmse']),
                'r2': float(metrics['r2']),
                'within_20': float(metrics['within_20']),
                'within_50': float(metrics['within_50']),
                'within_100': float(metrics['within_100']),
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"    Model: {model_path}")
    print(f"    Features: {feature_map_path}\n")

    # Summary
    print("=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Run analyze_xgboost_results.py for SHAP analysis")


if __name__ == "__main__":
    main()
