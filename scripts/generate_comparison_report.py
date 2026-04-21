#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère un rapport détaillé de comparaison entre les trois approches.
"""

import json
from typing import List, Dict, Tuple


def calculate_overlap(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
    """Calculer le pourcentage de chevauchement entre deux plages."""
    overlap_start = max(range1[0], range2[0])
    overlap_end = min(range1[1], range2[1])
    if overlap_start >= overlap_end:
        return 0.0
    overlap = overlap_end - overlap_start
    range1_len = range1[1] - range1[0]
    return (overlap / range1_len) * 100 if range1_len > 0 else 0.0


def load_all_data():
    """Charger toutes les données."""
    with open('data/processed/kitab_uqala_reference_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    with open('data/processed/kitab_uqala_boundaries.json', 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)

    with open('results/camelbert_kitab_uqala_segments.json', 'r', encoding='utf-8') as f:
        camelbert_data = json.load(f)
    camelbert_segments = camelbert_data.get('segmentation', {}).get('segments', [])

    return text, gold_standard, camelbert_segments


def generate_comparison_report(text, gold_standard, camelbert_segments):
    """Générer un rapport de comparaison détaillé."""

    report = {
        'metadata': {
            'corpus_size': len(text),
            'corpus_size_mb': round(len(text) / 1024 / 1024, 2),
            'gold_standard_count': len(gold_standard),
            'camelbert_segments': len(camelbert_segments),
        },
        'gold_standard_analysis': {
            'total_boundaries': len(gold_standard),
            'total_chars': sum(b['end'] - b['start'] for b in gold_standard),
            'avg_length': sum(b['end'] - b['start'] for b in gold_standard) / len(gold_standard) if gold_standard else 0,
            'min_length': min(b['end'] - b['start'] for b in gold_standard) if gold_standard else 0,
            'max_length': max(b['end'] - b['start'] for b in gold_standard) if gold_standard else 0,
        },
        'camelbert_analysis': {
            'total_segments': len(camelbert_segments),
            'isnads': len([s for s in camelbert_segments if s.get('type') == 'isnad']),
            'prose': len([s for s in camelbert_segments if s.get('type') == 'prose']),
            'total_chars': sum(s['end'] - s['start'] for s in camelbert_segments),
            'isnad_chars': sum(s['end'] - s['start'] for s in camelbert_segments if s.get('type') == 'isnad'),
        },
        'overlap_analysis': {},
    }

    # Analyser les chevauchements des isnads CAMeL-BERT avec le gold standard
    camelbert_isnads = [s for s in camelbert_segments if s.get('type') == 'isnad']
    gold_ranges = [(b['start'], b['end']) for b in gold_standard]

    overlap_percentages = []
    for isnad in camelbert_isnads:
        isnad_range = (isnad['start'], isnad['end'])
        # Calculer le chevauchement avec chaque gold boundary
        max_overlap = 0
        for gold_range in gold_ranges:
            overlap = calculate_overlap(isnad_range, gold_range)
            max_overlap = max(max_overlap, overlap)
        overlap_percentages.append(max_overlap)

    if overlap_percentages:
        report['overlap_analysis']['isnad_gold_overlap'] = {
            'avg': round(sum(overlap_percentages) / len(overlap_percentages), 2),
            'min': round(min(overlap_percentages), 2),
            'max': round(max(overlap_percentages), 2),
            'median': round(sorted(overlap_percentages)[len(overlap_percentages) // 2], 2),
            'perfect_overlaps_count': len([p for p in overlap_percentages if p >= 90]),
            'perfect_overlaps_pct': round(len([p for p in overlap_percentages if p >= 90]) / len(overlap_percentages) * 100, 2),
        }

    # Analyser la distribution des types
    report['type_distribution'] = {
        'isnads_percent': round(report['camelbert_analysis']['isnads'] / report['camelbert_analysis']['total_segments'] * 100, 2),
        'prose_percent': round(report['camelbert_analysis']['prose'] / report['camelbert_analysis']['total_segments'] * 100, 2),
        'isnad_chars_percent': round(report['camelbert_analysis']['isnad_chars'] / report['camelbert_analysis']['total_chars'] * 100, 2),
    }

    # Ajouter des samples de discordances
    report['sample_discrepancies'] = []

    # Chercher les segments mal alignés (chevauchement < 50%)
    for i, isnad in enumerate(camelbert_isnads[:5]):  # Premier 5
        isnad_range = (isnad['start'], isnad['end'])
        best_overlap = 0
        best_gold = None
        for gold_idx, gold_range in enumerate(gold_ranges):
            overlap = calculate_overlap(isnad_range, gold_range)
            if overlap > best_overlap:
                best_overlap = overlap
                best_gold = (gold_idx, gold_range)

        sample = {
            'camelbert_index': i,
            'camelbert_start': isnad['start'],
            'camelbert_end': isnad['end'],
            'camelbert_length': isnad['end'] - isnad['start'],
            'camelbert_text': text[isnad['start']:isnad['start'] + 50].strip(),
            'best_gold_match': {
                'index': best_gold[0] if best_gold else None,
                'overlap_percent': best_overlap,
            }
        }
        report['sample_discrepancies'].append(sample)

    return report


def main():
    print("Chargement des données...")
    text, gold_standard, camelbert_segments = load_all_data()

    print("Génération du rapport de comparaison...")
    report = generate_comparison_report(text, gold_standard, camelbert_segments)

    output_path = 'results/comparison_report_detailed.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nRapport généré: {output_path}")
    print(f"\nStatistiques Clés:")
    print(f"  Corpus size: {report['metadata']['corpus_size']:,} caractères")
    print(f"  Gold Standard: {report['metadata']['gold_standard_count']} boundaries")
    print(f"  CAMeL-BERT: {report['metadata']['camelbert_segments']} segments")
    print(f"    - Isnads: {report['camelbert_analysis']['isnads']}")
    print(f"    - Prose: {report['camelbert_analysis']['prose']}")
    print(f"\nChevauchement Gold↔CAMeL-BERT:")
    if 'isnad_gold_overlap' in report['overlap_analysis']:
        overlap = report['overlap_analysis']['isnad_gold_overlap']
        print(f"  Moyenne: {overlap['avg']}%")
        print(f"  Médiane: {overlap['median']}%")
        print(f"  Parfaits (≥90%): {overlap['perfect_overlaps_count']} / {report['camelbert_analysis']['isnads']} ({overlap['perfect_overlaps_pct']}%)")


if __name__ == "__main__":
    main()
