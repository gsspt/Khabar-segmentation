#!/usr/bin/env python3
"""
Compare trois approches de segmentation :
1. Baseline v4 (rule-based)
2. CAMeL-BERT (fine-tuned)
3. Gold Standard (Deepseek API)

Métriques :
- Détection d'akhbars (recall, precision, F1)
- Exactitude des boundaries
- Confiance des prédictions
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np


def load_gold_standard(path: Path) -> Dict:
    """Charger le gold standard généré par Deepseek."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def segment_with_baseline_v4(text: str) -> List[Dict]:
    """
    Segmenter avec baseline v4.

    Note: Cette fonction doit importer et utiliser baseline_v4.py
    Pour maintenant, on retourne une structure compatible.
    """
    # TODO: Importer segment() depuis scripts/baselines/baseline_v4.py
    # Placeholder pour l'instant
    return []


def parse_boundaries_from_gold_standard(gold_data: Dict) -> List[Dict]:
    """
    Parse le gold standard et génère des boundaries.

    Retourne une liste de boundaries avec positions estimées.
    """
    # On reconstruit le texte et les positions
    boundaries = []
    for akh in gold_data['akhbars']:
        boundary = {
            'id': akh.get('id'),
            'type': akh.get('type'),
            'confidence': akh.get('confidence'),
            'content_length': len(akh.get('content', akh.get('isnad', ''))) + len(akh.get('content', '')),
        }
        boundaries.append(boundary)
    return boundaries


def evaluate_predictions(gold_boundaries: List[Dict], pred_boundaries: List[Dict]) -> Dict:
    """
    Comparer les prédictions avec le gold standard.

    Métriques :
    - Nombre de boundaries détectés
    - Exactitude des positions
    - Confiance moyenne
    """
    gold_count = len(gold_boundaries)
    pred_count = len(pred_boundaries)

    # Calculs simples pour l'instant
    metrics = {
        'gold_count': gold_count,
        'predicted_count': pred_count,
        'detection_recall': pred_count / gold_count if gold_count > 0 else 0,
        'avg_confidence': np.mean([b['confidence'] for b in gold_boundaries]) if gold_boundaries else 0,
    }

    return metrics


def main():
    # Chemins
    gold_file = Path('data/gold_standard/gold_standard_0392IbnIsmacil.json')

    if not gold_file.exists():
        print(f"[ERROR] Gold standard file not found: {gold_file}")
        print(f"[INFO] Run create_gold_standard_deepseek.py first")
        return

    print(f"[INFO] Loading gold standard...")
    gold_data = load_gold_standard(gold_file)

    print(f"\n[GOLD STANDARD]")
    print(f"  File: {gold_data['file']}")
    print(f"  Model: {gold_data['model']}")
    print(f"  Total akhbars: {len(gold_data['akhbars'])}")

    # Analyse par type
    type_counts = defaultdict(int)
    confidence_by_type = defaultdict(list)

    for akh in gold_data['akhbars']:
        seg_type = akh.get('type', 'unknown')
        type_counts[seg_type] += 1
        confidence_by_type[seg_type].append(akh.get('confidence', 0))

    print(f"\n[SEGMENT TYPES]")
    for seg_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        avg_conf = np.mean(confidence_by_type[seg_type])
        print(f"  {seg_type:<20} : {count:3d} segments (avg confidence: {avg_conf:.2f})")

    # TODO: Intégrer baseline_v4 et CAMeL-BERT
    # Pour maintenant, afficher juste les stats du gold standard

    print(f"\n[STATUS]")
    print(f"  Gold standard generated: OK")
    print(f"  Baseline v4 integration: TODO")
    print(f"  CAMeL-BERT integration: TODO")
    print(f"\n[NEXT STEPS]")
    print(f"  1. Implement baseline_v4 segmentation")
    print(f"  2. Integrate CAMeL-BERT inference")
    print(f"  3. Compare metrics across all three")


if __name__ == '__main__':
    main()
