#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP analysis for XGBoost boundary model.

Analyze feature importance and model interpretability using SHAP values.
Provides insights into which features drive boundary predictions.
"""

import json
from pathlib import Path
from typing import Tuple
import numpy as np

try:
    import xgboost as xgb
    import shap
    from sklearn.model_selection import train_test_split
except ImportError:
    print("[ERROR] Missing required libraries. Install with:")
    print("  pip install xgboost shap numpy scikit-learn")
    exit(1)


def load_model_and_features(model_path: Path, feature_names_path: Path) -> Tuple:
    """Load trained model and feature metadata."""
    model = xgb.Booster()
    model.load_model(str(model_path))

    with open(feature_names_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return model, metadata


def load_features_from_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load features and target from CSV."""
    import csv

    features_list = []
    targets = []
    feature_names = None

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        feature_names = [col for col in reader.fieldnames if col != 'target_offset']

        for row in reader:
            feature_values = [float(row.get(col, 0)) for col in feature_names]
            features_list.append(feature_values)

            target = float(row.get('target_offset', 0))
            targets.append(target)

    X = np.array(features_list)
    y = np.array(targets)

    return X, y, feature_names


def compute_shap_values(model: xgb.Booster, X_data: np.ndarray,
                       feature_names: list, sample_size: int = 100) -> Tuple:
    """
    Compute SHAP values for interpretability.

    Args:
        model: Trained XGBoost model
        X_data: Feature matrix
        feature_names: List of feature names
        sample_size: Number of samples to use (for speed)

    Returns:
        (shap_values, background_data, explainer)
    """
    print("[*] Computing SHAP values (this may take a moment)...")

    # Use subset for speed
    sample_idx = np.random.choice(len(X_data), min(sample_size, len(X_data)), replace=False)
    X_sample = X_data[sample_idx]

    # Create DMatrix
    dmatrix = xgb.DMatrix(X_data, feature_names=feature_names)
    background = xgb.DMatrix(X_sample, feature_names=feature_names)

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(background)

    return shap_values, background, explainer


def generate_shap_report(shap_values: np.ndarray, feature_names: list,
                        output_path: Path):
    """Generate a text report of SHAP feature importance."""
    print("[*] Generating SHAP report...")

    # Mean absolute SHAP values (global feature importance)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# SHAP Feature Importance Analysis\n\n")
        f.write("## Global Feature Importance (Mean |SHAP|)\n\n")
        f.write("Feature importance shows how much each feature contributes to model predictions.\n")
        f.write("Higher values = more important for determining isnad boundaries.\n\n")

        f.write("| Rank | Feature | Importance |\n")
        f.write("|------|---------|-------------|\n")

        for rank, (feat, importance) in enumerate(feature_importance, 1):
            f.write(f"| {rank} | {feat} | {importance:.4f} |\n")

        f.write("\n## Interpretation\n\n")

        f.write("### Top Features Explained\n\n")

        # Provide interpretation for top features
        interpretations = {
            'has_qal': "Whether 'قال' marker exists (primary boundary signal)",
            'qal_distance': "Distance to 'قال' from isnad start (direct position hint)",
            'pronoun_count': "Count of first-person pronouns (typical in isnads)",
            'narrative_verb_count': "Count of narrative verbs (marks khabar beginning)",
            'distance_to_next_isnad': "Distance to next isnads (bounds the boundary)",
            'window_char_count': "Total characters in search window (size indicator)",
            'qal_to_reference_diff': "Difference between 'قال' and true boundary",
            'window_word_count': "Total words in search window",
        }

        for i, (feat, _) in enumerate(feature_importance[:5]):
            explanation = interpretations.get(feat, "Feature indicates boundary characteristics")
            f.write(f"**{i+1}. {feat}**: {explanation}\n\n")

    print(f"    [OK] Report saved: {output_path}")


def main():
    """Run SHAP analysis on trained XGBoost model."""
    print("=" * 80)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80 + "\n")

    # Check files
    model_path = Path("../results/xgboost_boundary_model.json")
    feature_map_path = Path("../results/xgboost_feature_names.json")
    features_csv_path = Path("../data/processed/xgboost_training_features.csv")

    if not model_path.exists():
        print("[ERROR] Model not found. Run xgboost_boundary_model.py first.")
        exit(1)

    # Load model and metadata
    print("[1] Loading model and metadata...")
    model, metadata = load_model_and_features(model_path, feature_map_path)
    feature_names = metadata['feature_names']
    print(f"    Features: {metadata['feature_count']}")
    print(f"    Metrics: MAE={metadata['metrics']['mae']:.1f}, Within-50={metadata['metrics']['within_50']:.1f}%\n")

    # Load features
    print("[2] Loading feature data...")
    X, y, loaded_feature_names = load_features_from_csv(features_csv_path)
    print(f"    Samples: {len(X)}\n")

    # Compute SHAP values
    print("[3] Computing SHAP values...")
    shap_values, background, explainer = compute_shap_values(model, X, feature_names)
    print(f"    [OK] SHAP values computed\n")

    # Generate report
    print("[4] Generating report...")
    report_path = Path("../results/xgboost_shap_analysis.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    generate_shap_report(shap_values, feature_names, report_path)

    # Summary
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nReport: {report_path}")
    print("\nKey Findings:")
    print("- Top features drive boundary predictions")
    print("- SHAP explains model's decision-making process")
    print("- Use these insights to engineer better features or refine linguistic rules")


if __name__ == "__main__":
    main()
