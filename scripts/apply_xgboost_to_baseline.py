#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply trained XGBoost model to v3.5 baseline boundaries.

Use the trained model to refine isnad end positions detected by v3.5.
Compare boundary precision improvements.
"""

import re
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    print("[ERROR] XGBoost not installed. Run: pip install xgboost")
    exit(1)

# Import feature extraction functions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from xgboost_feature_extraction import (
    normalize_arabic_text, extract_features, NARRATIVE_VERBS
)


def load_baseline_results():
    """Load v3.5 baseline results."""
    from baseline_v3_5_improved import segment_akhbars_v3_5

    with open("../data/processed/kitab_uqala_reference_corpus.txt", 'r', encoding='utf-8') as f:
        corpus = f.read()

    akhbars = segment_akhbars_v3_5(corpus)
    return corpus, akhbars


def load_xgboost_model():
    """Load trained XGBoost model and feature metadata."""
    model = xgb.Booster()
    model.load_model(str(Path("../results/xgboost_boundary_model.json")))

    with open("../results/xgboost_feature_names.json", 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return model, metadata


def predict_isnad_end_with_xgboost(model, features: Dict, feature_names: list) -> float:
    """
    Use XGBoost to predict isnad end offset.

    Args:
        model: Trained XGBoost model
        features: Feature dict from extract_features()
        feature_names: List of feature names in order

    Returns:
        Predicted offset (chars from isnad start to end)
    """
    # Build feature vector in correct order
    feature_vector = np.array([[features.get(name, 0) for name in feature_names]])

    # Predict
    dmatrix = xgb.DMatrix(feature_vector, feature_names=feature_names)
    prediction = model.predict(dmatrix)[0]

    return prediction


def apply_xgboost_refinement(corpus: str, baseline_akhbars: List[Dict],
                             model, metadata: Dict) -> List[Dict]:
    """
    Apply XGBoost refinement to v3.5 baseline results.

    Args:
        corpus: Full text
        baseline_akhbars: Akhbars detected by v3.5
        model: Trained XGBoost model
        metadata: Model metadata (feature names, etc.)

    Returns:
        Refined akhbars with XGBoost-predicted boundaries
    """
    feature_names = metadata['feature_names']
    refined_akhbars = []

    for i, akhbar in enumerate(baseline_akhbars):
        isnad_start = akhbar['start']
        khabar_end = akhbar['end']

        # Find next akhbar start
        next_isnad_start = len(corpus)
        if i + 1 < len(baseline_akhbars):
            next_isnad_start = baseline_akhbars[i + 1]['start']

        # Extract features
        try:
            features = extract_features(
                corpus,
                isnad_start,
                akhbar['isnad_end'],  # Use baseline as reference for feature extraction
                next_isnad_start,
            )

            # Predict isnad end offset
            predicted_offset = predict_isnad_end_with_xgboost(model, features, feature_names)

            # Bound the prediction
            predicted_offset = max(20, min(predicted_offset, next_isnad_start - isnad_start - 20))

            # New isnad end
            refined_isnad_end = isnad_start + predicted_offset

            # Create refined akhbar
            refined_akhbar = akhbar.copy()
            refined_akhbar['isnad_end_xgboost'] = int(refined_isnad_end)
            refined_akhbar['isnad_end_baseline'] = akhbar['isnad_end']
            refined_akhbar['offset_change'] = int(refined_isnad_end - akhbar['isnad_end'])

            refined_akhbars.append(refined_akhbar)

        except Exception as e:
            # Keep baseline if extraction fails
            refined_akhbars.append(akhbar)

    return refined_akhbars


def compute_boundary_metrics(corpus: str, detected_boundaries: List[Dict],
                            reference_boundaries: List[Dict],
                            use_xgboost: bool = False) -> Dict:
    """
    Compute boundary precision metrics.

    Args:
        corpus: Full text
        detected_boundaries: Detected akhbars
        reference_boundaries: Reference boundaries from JSON
        use_xgboost: If True, use XGBoost-refined end positions

    Returns:
        Dict of metrics
    """
    start_deviations = []
    end_deviations = []
    iou_scores = []

    for detected in detected_boundaries:
        # Find closest reference boundary
        det_start = detected['start']
        det_end = detected['end']

        if use_xgboost:
            # Use XGBoost-refined isnad end
            det_isnad_end = detected.get('isnad_end_xgboost', detected['isnad_end'])
        else:
            det_isnad_end = detected['isnad_end']

        # Find reference boundary close to this
        best_match = None
        best_dist = float('inf')

        for ref in reference_boundaries:
            dist = abs(det_start - ref['start'])
            if dist < best_dist:
                best_dist = dist
                best_match = ref

        if best_match is None:
            continue

        # Metrics
        ref_start = best_match['start']
        ref_end = best_match['end']

        start_dev = abs(det_start - ref_start)
        end_dev = abs(det_end - ref_end)

        start_deviations.append(start_dev)
        end_deviations.append(end_dev)

        # IoU (Jaccard similarity)
        intersection = max(0, min(det_end, ref_end) - max(det_start, ref_start))
        union = max(det_end, ref_end) - min(det_start, ref_start)
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou)

    return {
        'start_mean': np.mean(start_deviations) if start_deviations else 0,
        'start_median': np.median(start_deviations) if start_deviations else 0,
        'end_mean': np.mean(end_deviations) if end_deviations else 0,
        'end_median': np.median(end_deviations) if end_deviations else 0,
        'iou_mean': np.mean(iou_scores) if iou_scores else 0,
        'iou_median': np.median(iou_scores) if iou_scores else 0,
        'usable_pct': (np.sum(np.array(iou_scores) >= 0.80) / len(iou_scores) * 100) if iou_scores else 0,
        'count': len(start_deviations),
    }


def main():
    """Apply XGBoost refinement and evaluate."""
    print("=" * 80)
    print("APPLYING XGBOOST REFINEMENT TO V3.5 BASELINE")
    print("=" * 80 + "\n")

    # Load data
    print("[1] Loading baseline results...")
    corpus, baseline_akhbars = load_baseline_results()

    with open("../data/processed/kitab_uqala_boundaries.json", 'r', encoding='utf-8') as f:
        reference_boundaries = json.load(f)

    print(f"    Baseline detected: {len(baseline_akhbars)}")
    print(f"    Reference: {len(reference_boundaries)}\n")

    # Load model
    print("[2] Loading trained XGBoost model...")
    model, metadata = load_xgboost_model()
    print(f"    Model loaded\n")

    # Apply refinement
    print("[3] Applying XGBoost refinement...")
    refined_akhbars = apply_xgboost_refinement(corpus, baseline_akhbars, model, metadata)
    print(f"    Refined: {len(refined_akhbars)}\n")

    # Evaluate baseline
    print("[4] Evaluating baseline (v3.5)...")
    baseline_metrics = compute_boundary_metrics(corpus, baseline_akhbars, reference_boundaries, use_xgboost=False)
    print(f"    Start boundary error: {baseline_metrics['start_mean']:.0f} chars (median {baseline_metrics['start_median']:.0f})")
    print(f"    End boundary error: {baseline_metrics['end_mean']:.0f} chars (median {baseline_metrics['end_median']:.0f})")
    print(f"    Mean IoU: {baseline_metrics['iou_mean']:.3f}")
    print(f"    Usable (80%+ IoU): {baseline_metrics['usable_pct']:.1f}%\n")

    # Evaluate refined
    print("[5] Evaluating refined (v3.5 + XGBoost)...")
    refined_metrics = compute_boundary_metrics(corpus, refined_akhbars, reference_boundaries, use_xgboost=True)
    print(f"    Start boundary error: {refined_metrics['start_mean']:.0f} chars (median {refined_metrics['start_median']:.0f})")
    print(f"    End boundary error: {refined_metrics['end_mean']:.0f} chars (median {refined_metrics['end_median']:.0f})")
    print(f"    Mean IoU: {refined_metrics['iou_mean']:.3f}")
    print(f"    Usable (80%+ IoU): {refined_metrics['usable_pct']:.1f}%\n")

    # Compute improvements
    print("[6] Improvements:")
    start_improve = ((baseline_metrics['start_mean'] - refined_metrics['start_mean']) / baseline_metrics['start_mean'] * 100) if baseline_metrics['start_mean'] > 0 else 0
    end_improve = ((baseline_metrics['end_mean'] - refined_metrics['end_mean']) / baseline_metrics['end_mean'] * 100) if baseline_metrics['end_mean'] > 0 else 0
    iou_improve = ((refined_metrics['iou_mean'] - baseline_metrics['iou_mean']) / baseline_metrics['iou_mean'] * 100) if baseline_metrics['iou_mean'] > 0 else 0
    usable_improve = refined_metrics['usable_pct'] - baseline_metrics['usable_pct']

    print(f"    Start error: {start_improve:+.1f}%")
    print(f"    End error: {end_improve:+.1f}%")
    print(f"    IoU improvement: {iou_improve:+.1f}%")
    print(f"    Usable increase: {usable_improve:+.1f} percentage points\n")

    # Save results
    print("[7] Saving refined results...")
    output_path = Path("../results/xgboost_refined_boundaries.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(refined_akhbars, f, indent=2, ensure_ascii=False)

    # Save comparison report
    report_path = Path("../results/xgboost_evaluation_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# XGBoost Boundary Refinement Evaluation\n\n")
        f.write("## Baseline (v3.5)\n\n")
        f.write(f"- Start error: {baseline_metrics['start_mean']:.0f} chars (median {baseline_metrics['start_median']:.0f})\n")
        f.write(f"- End error: {baseline_metrics['end_mean']:.0f} chars (median {baseline_metrics['end_median']:.0f})\n")
        f.write(f"- Mean IoU: {baseline_metrics['iou_mean']:.3f}\n")
        f.write(f"- Usable (80%+): {baseline_metrics['usable_pct']:.1f}%\n\n")

        f.write("## Refined (v3.5 + XGBoost)\n\n")
        f.write(f"- Start error: {refined_metrics['start_mean']:.0f} chars (median {refined_metrics['start_median']:.0f})\n")
        f.write(f"- End error: {refined_metrics['end_mean']:.0f} chars (median {refined_metrics['end_median']:.0f})\n")
        f.write(f"- Mean IoU: {refined_metrics['iou_mean']:.3f}\n")
        f.write(f"- Usable (80%+): {refined_metrics['usable_pct']:.1f}%\n\n")

        f.write("## Improvements\n\n")
        f.write(f"- Start error: {start_improve:+.1f}%\n")
        f.write(f"- End error: {end_improve:+.1f}%\n")
        f.write(f"- IoU improvement: {iou_improve:+.1f}%\n")
        f.write(f"- Usable increase: {usable_improve:+.1f} percentage points\n\n")

        f.write("## Conclusion\n\n")
        if usable_improve > 10:
            f.write("XGBoost refinement provides significant boundary improvement.\n")
        elif usable_improve > 0:
            f.write("XGBoost refinement provides modest boundary improvement.\n")
        else:
            f.write("XGBoost refinement does not improve boundaries substantially.\n")

    print(f"    Report: {report_path}")
    print(f"    Results: {output_path}\n")

    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
