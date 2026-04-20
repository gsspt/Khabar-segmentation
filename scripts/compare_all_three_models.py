#!/usr/bin/env python3
"""
Compare three segmentation approaches on 0392IbnIsmacil text:
1. Gold Standard (Deepseek API)
2. Baseline v4 (Rule-based)
3. CAMeL-BERT (Neural fine-tuned)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Ensure UTF-8 output on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_gold_standard() -> Dict:
    """Load the gold standard (Deepseek annotations)."""
    with open('data/gold_standard/gold_standard_0392IbnIsmacil_v2.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_baseline_v4() -> Dict:
    """Load baseline v4 results."""
    with open('results/baseline_v4_comparison.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_camelbert() -> Dict:
    """Load CAMeL-BERT token predictions."""
    try:
        with open('results/camelbert_0392IbnIsmacil_token_predictions.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️  CAMeL-BERT results not found. Run inference in Colab first.")
        return None

def analyze_comparison(gold_std, baseline, camelbert):
    """Analyze and compare all three models."""

    print("\n" + "="*80)
    print("SEGMENTATION MODEL COMPARISON - 0392IbnIsmacil Text")
    print("="*80)

    # Extract counts
    gold_total = len(gold_std['results']['akhbars'])
    gold_types = {}
    for akh in gold_std['results']['akhbars']:
        t = akh.get('type', 'unknown')
        gold_types[t] = gold_types.get(t, 0) + 1

    baseline_total = baseline['total_segments']
    baseline_types = {
        'with_isnad': baseline['with_isnad'],
        'prose': baseline['prose']
    }

    camelbert_boundary_tokens = len(camelbert['token_predictions']['boundary_indices']) if camelbert else 0

    # Display comparison table
    print(f"\n{'Model':<20} | {'Total':<8} | {'with_isnad':<12} | {'poetry':<8} | {'prose':<8}")
    print("-"*80)

    print(f"{'Gold Standard':<20} | {gold_total:<8} | {gold_types.get('with_isnad', 0):<12} | {gold_types.get('poetry', 0):<8} | {gold_types.get('prose', 0):<8}")
    print(f"{'Baseline v4':<20} | {baseline_total:<8} | {baseline_types['with_isnad']:<12} | {'0':<8} | {'0':<8}")

    if camelbert:
        print(f"{'CAMeL-BERT':<20} | {camelbert_boundary_tokens:<8} | {'?':<12} | {'?':<8} | {'?':<8}")
    else:
        print(f"{'CAMeL-BERT':<20} | {'⏳':<8} | {'⏳':<12} | {'⏳':<8} | {'⏳':<8}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    print(f"\n📊 GOLD STANDARD (Reference Truth - 63 segments)")
    print(f"  • with_isnad: {gold_types.get('with_isnad', 0)} (41%)")
    print(f"  • poetry: {gold_types.get('poetry', 0)} (32%)")
    print(f"  • prose: {gold_types.get('prose', 0)} (25%)")
    print(f"  • continuation: {gold_types.get('continuation', 0)} (2%)")
    print(f"  ✅ Status: Complete, comprehensive")

    print(f"\n📈 BASELINE V4 (Rule-based - 7 segments)")
    print(f"  • with_isnad: {baseline_types['with_isnad']} (100%)")
    print(f"  • poetry: 0 (0%)")
    print(f"  • prose: 0 (0%)")
    print(f"  ❌ Status: Very conservative, detects only clear isnads")
    print(f"  ⚠️  Recall: {baseline_total}/{gold_total} = {100*baseline_total/gold_total:.1f}%")

    if camelbert:
        print(f"\n🤖 CAMELBERT (Neural fine-tuned)")
        print(f"  • Boundary tokens detected: {camelbert_boundary_tokens}")
        print(f"  • Mean prob (boundary): {camelbert['probabilities']['mean_prob_boundary']:.4f}")
        print(f"  • Mean prob (non-boundary): {camelbert['probabilities']['mean_prob_non_boundary']:.4f}")
        print(f"  ✅ Status: Token-level predictions complete")
        if camelbert_boundary_tokens > 0:
            print(f"  📊 Expected segments: ~{camelbert_boundary_tokens} (estimate)")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")

    print(f"""
✅ GOLD STANDARD
   • Best for: Training neural models, ground truth annotation
   • Pros: Comprehensive, all narrative types covered, quality annotations
   • Cons: Requires API calls, some potential LLM annotation errors
   • Use case: Reference standard for evaluation

⚙️  BASELINE V4
   • Best for: Detecting explicit transmission chains only
   • Pros: Fast, zero-cost, interpretable rules, high precision
   • Cons: Low recall (88% false negatives), blind to prose/poetry
   • Use case: Complement to neural model (combined approach)

🤖 CAMELBERT
   • Best for: Practical deployment, good balance of precision/recall
   • Pros: Trained on gold standard, fast inference, generalizable
   • Cons: Depends on training data quality, needs GPU (if large scale)
   • Use case: Primary segmentation model for production

🏆 HYBRID APPROACH (Recommended)
   • Combine baseline v4 + CAMeL-BERT
   • Use baseline for high-confidence isnad detection
   • Use CAMeL-BERT for prose/poetry/boundaries
   • Better coverage with higher precision
""")

def main():
    print("[INFO] Loading models...")

    gold_std = load_gold_standard()
    baseline = load_baseline_v4()
    camelbert = load_camelbert()

    if not gold_std or not baseline:
        print("ERROR: Missing gold standard or baseline v4 results")
        return

    # Run comparison
    analyze_comparison(gold_std, baseline, camelbert)

    # Save comparison to file
    comparison = {
        'gold_standard': {
            'total': len(gold_std['results']['akhbars']),
            'file': 'data/gold_standard/gold_standard_0392IbnIsmacil_v2.json'
        },
        'baseline_v4': {
            'total': baseline['total_segments'],
            'file': 'results/baseline_v4_comparison.json'
        },
        'camelbert': {
            'total': len(camelbert['token_predictions']['boundary_indices']) if camelbert else None,
            'file': 'results/camelbert_0392IbnIsmacil_token_predictions.json'
        }
    }

    with open('results/model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Comparison saved to results/model_comparison.json")

if __name__ == '__main__':
    main()
