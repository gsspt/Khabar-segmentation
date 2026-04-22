#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation déterministe au niveau des ISNADS
Comparer les isnads du gold standard avec les clusters CAMeL-BERT
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_corpus() -> str:
    """Charger le texte brut du corpus."""
    with open('data/processed/kitab_uqala_reference_corpus.txt', 'r', encoding='utf-8') as f:
        return f.read()


def extract_isnads_from_gold_standard(text: str) -> List[Dict]:
    """
    Extraire les isnads du gold standard avec leurs positions.
    """
    print("\n[1/4] Extraire les isnads du gold standard...")

    with open('data/processed/kitab_uqala_al_majanin_annotated.json', 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    isnads = []
    akhbars_with_isnads = 0
    akhbars_without_isnads = 0

    for akhbar in gold_data['akhbar']:
        akhbar_num = akhbar['num']
        segments = akhbar.get('content', {}).get('segments', [])

        # Chercher le segment isnad
        isnad_text = None
        for seg in segments:
            if seg.get('type') == 'isnad':
                isnad_text = seg.get('text', '').strip()
                break

        if not isnad_text:
            akhbars_without_isnads += 1
            continue

        # Chercher ce texte dans le corpus
        pos = text.find(isnad_text)
        if pos != -1:
            isnads.append({
                'akhbar_num': akhbar_num,
                'isnad_text_first_50': isnad_text[:50],
                'isnad_full_text': isnad_text,
                'isnad_start': pos,
                'isnad_end': pos + len(isnad_text),
                'isnad_length': len(isnad_text),
            })
            akhbars_with_isnads += 1

    print(f"      Akhbars avec isnads trouvés: {akhbars_with_isnads}")
    print(f"      Akhbars sans isnads: {akhbars_without_isnads}")
    print(f"      Total isnads: {len(isnads)}")

    return isnads


def load_camelbert_clusters() -> List[Dict]:
    """
    Charger les clusters CAMeL-BERT depuis le fichier pré-calculé.
    """
    print("\n[2/4] Charger les clusters CAMeL-BERT...")

    with open('results/camelbert_char_boundaries_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    clusters = []
    for boundary in data.get('khabar_boundaries', []):
        clusters.append({
            'boundary_id': boundary['boundary_id'],
            'cluster_start': boundary['char_start'],
            'cluster_end': boundary['char_end'],
            'n_tokens': boundary['n_tokens'],
        })

    metadata = data.get('metadata', {})
    print(f"      Clusters trouvés: {len(clusters)}")
    print(f"      F1 vs gold (tol ±80): {metadata.get('evaluation_tol80', {}).get('f1', 'N/A')}")

    return clusters, metadata


def calculate_overlap(seg1_start: int, seg1_end: int, seg2_start: int, seg2_end: int) -> float:
    """Calculer le pourcentage de chevauchement."""
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap = overlap_end - overlap_start
    seg1_length = seg1_end - seg1_start

    return (overlap / seg1_length) * 100 if seg1_length > 0 else 0.0


def validate_isnads_vs_clusters(isnads: List[Dict], clusters: List[Dict]) -> Dict:
    """
    Valider que CAMeL-BERT détecte bien les isnads du gold standard.
    """
    print("\n[3/4] Valider les isnads...")

    print("\n" + "="*80)
    print("VALIDATION: ISNADS GOLD STANDARD vs CLUSTERS CAMeL-BERT")
    print("="*80)

    print(f"\nGold Standard:")
    print(f"  Isnads: {len(isnads)}")

    print(f"\nCAMeL-BERT:")
    print(f"  Clusters: {len(clusters)}")

    # Pour chaque isnad, trouver le meilleur cluster correspondant
    print(f"\n" + "-"*80)
    print("ANALYSE: Isnads Gold → Clusters CAMeL-BERT")
    print("-"*80)

    matches = []

    for isnad in isnads:
        best_overlap = 0
        best_cluster_idx = -1
        best_distance = float('inf')

        for cluster_idx, cluster in enumerate(clusters):
            # Calculer l'overlap
            overlap = calculate_overlap(
                isnad['isnad_start'], isnad['isnad_end'],
                cluster['cluster_start'], cluster['cluster_end']
            )

            # Calculer la distance entre les débuts
            distance = abs(cluster['cluster_start'] - isnad['isnad_start'])

            if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
                best_overlap = overlap
                best_cluster_idx = cluster_idx
                best_distance = distance

        matches.append({
            'akhbar_num': isnad['akhbar_num'],
            'gold_start': isnad['isnad_start'],
            'gold_end': isnad['isnad_end'],
            'gold_length': isnad['isnad_length'],
            'best_cluster_idx': best_cluster_idx,
            'best_overlap': best_overlap,
            'best_distance': best_distance if best_cluster_idx != -1 else None,
            'isnad_sample': isnad['isnad_text_first_50'],
        })

    # Statistiques
    perfect_matches = sum(1 for m in matches if m['best_overlap'] >= 90)
    good_matches = sum(1 for m in matches if 70 <= m['best_overlap'] < 90)
    acceptable_matches = sum(1 for m in matches if 50 <= m['best_overlap'] < 70)
    bad_matches = sum(1 for m in matches if m['best_overlap'] < 50)

    if matches:
        avg_overlap = sum(m['best_overlap'] for m in matches) / len(matches)
        distances = [m['best_distance'] for m in matches if m['best_distance'] is not None]
        if distances:
            distances.sort()
            median_distance = distances[len(distances) // 2]
        else:
            median_distance = None
    else:
        avg_overlap = 0
        median_distance = None

    print(f"\nCouverture des isnads:")
    print(f"  Parfaits (≥90% overlap): {perfect_matches}/{len(matches)} ({100*perfect_matches/len(matches):.1f}%)")
    print(f"  Bons (70-90%): {good_matches}/{len(matches)}")
    print(f"  Acceptables (50-70%): {acceptable_matches}/{len(matches)}")
    print(f"  Mauvais (<50%): {bad_matches}/{len(matches)}")
    print(f"\nOverlap moyen: {avg_overlap:.1f}%")
    if median_distance is not None:
        print(f"Distance médiane du début: {median_distance:.0f} chars")

    # Exemples
    print(f"\n" + "-"*80)
    print("EXEMPLES DE BONS MATCHES (≥90% overlap)")
    print("-"*80)

    good_examples = [m for m in matches if m['best_overlap'] >= 90][:5]
    for m in good_examples:
        print(f"\n  Akhbar #{m['akhbar_num']}:")
        print(f"    Gold:    [{m['gold_start']:6d}:{m['gold_end']:6d}]")
        if m['best_cluster_idx'] != -1:
            c = clusters[m['best_cluster_idx']]
            print(f"    Cluster: [{c['cluster_start']:6d}:{c['cluster_end']:6d}]")
        print(f"    Overlap: {m['best_overlap']:.1f}%")
        print(f"    Isnad: {m['isnad_sample']}...")

    # Isnads manqués
    print(f"\n" + "-"*80)
    print("ISNADS NON DÉTECTÉS (<50% overlap)")
    print("-"*80)

    bad_examples = [m for m in matches if m['best_overlap'] < 50]
    if bad_examples:
        print(f"  Total: {len(bad_examples)}/{len(matches)}")
        for m in bad_examples[:5]:
            print(f"\n  Akhbar #{m['akhbar_num']}:")
            print(f"    Gold: [{m['gold_start']:6d}:{m['gold_end']:6d}]")
            print(f"    Overlap: {m['best_overlap']:.1f}%")
            print(f"    Isnad: {m['isnad_sample']}...")
    else:
        print("  ✓ Tous les isnads sont détectés!")

    return {
        'matches': matches,
        'gold_count': len(isnads),
        'camelbert_count': len(clusters),
        'perfect_matches': perfect_matches,
        'good_matches': good_matches,
        'acceptable_matches': acceptable_matches,
        'bad_matches': bad_matches,
        'avg_overlap': avg_overlap,
        'median_distance': median_distance,
        'recall': perfect_matches / len(isnads) if isnads else 0,
    }


def save_results(validation: Dict):
    """Sauvegarder les résultats."""
    print("\n[4/4] Sauvegarder les résultats...")

    results = {
        'validation_date': '2026-04-22',
        'method': 'isnad_level_validation',
        'description': 'Validation au niveau des isnads: comparer les positions exactes des isnads du gold standard avec les clusters CAMeL-BERT',
        'gold_standard_count': validation['gold_count'],
        'camelbert_clusters_count': validation['camelbert_count'],
        'perfect_detection': validation['perfect_matches'],
        'detection_rate': round(validation['recall'] * 100, 1),
        'average_overlap': round(validation['avg_overlap'], 1),
        'median_distance_chars': validation['median_distance'],
        'statistics': {
            'perfect_matches': validation['perfect_matches'],
            'good_matches': validation['good_matches'],
            'acceptable_matches': validation['acceptable_matches'],
            'bad_matches': validation['bad_matches'],
        },
        'detailed_matches': validation['matches'],
    }

    output_path = 'results/validation_isnads_vs_clusters.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ Résultats sauvegardés: {output_path}")

    # Résumé
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"\nCAMeL-BERT détecte {validation['perfect_matches']}/{validation['gold_count']} isnads avec ≥90% overlap")
    print(f"Taux de détection: {validation['recall']*100:.1f}%")
    print(f"Overlap moyen: {validation['avg_overlap']:.1f}%")


def main():
    text = load_corpus()
    print(f"Corpus: {len(text):,} caractères")

    isnads = extract_isnads_from_gold_standard(text)
    clusters, metadata = load_camelbert_clusters()

    validation = validate_isnads_vs_clusters(isnads, clusters)
    save_results(validation)


if __name__ == "__main__":
    main()
