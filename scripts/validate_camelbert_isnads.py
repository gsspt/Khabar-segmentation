#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation déterministe de CAMeL-BERT vs gold standard
1. Extraire les positions exactes des isnads du gold standard
2. Extraire les positions des boundary tokens de CAMeL-BERT
3. Comparer directement les positions
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple


def load_corpus() -> str:
    """Charger le texte brut du corpus."""
    with open('data/processed/kitab_uqala_reference_corpus.txt', 'r', encoding='utf-8') as f:
        return f.read()


def extract_isnad_positions_from_gold_standard(text: str) -> List[Dict]:
    """
    Extraire les positions exactes des isnads du gold standard.

    Stratégie:
    1. Charger le JSON annoté
    2. Pour chaque akhbar avec un segment "isnad", extraire le texte
    3. Chercher ce texte dans le corpus brut et enregistrer start/end
    """
    print("\n[1/5] Extraire les positions des isnads du gold standard...")

    with open('data/processed/kitab_uqala_al_majanin_annotated.json', 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    isnads = []
    found = 0
    not_found = 0

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
            continue

        # Chercher ce texte dans le corpus
        pos = text.find(isnad_text)
        if pos != -1:
            isnads.append({
                'akhbar_num': akhbar_num,
                'isnad_text': isnad_text[:50],  # Afficher les 50 premiers chars
                'isnad_start': pos,
                'isnad_end': pos + len(isnad_text),
                'isnad_length': len(isnad_text),
            })
            found += 1
        else:
            not_found += 1

    print(f"      Isnads trouvés dans le corpus: {found}")
    print(f"      Isnads NON trouvés: {not_found}")
    print(f"      Total: {len(isnads)}")

    return isnads, gold_data


def load_camelbert_boundary_positions() -> List[Dict]:
    """
    Charger les positions des boundary tokens de CAMeL-BERT.

    Utiliser le fichier raw_inference qui contient les offsets globaux.
    """
    print("\n[2/5] Charger les boundary tokens de CAMeL-BERT...")

    with open('results/camelbert_kitab_uqala_raw_inference.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Extraire les arrays
    predictions = raw_data.get('inference_results', {}).get('predictions', [])
    offsets = raw_data.get('offsets', [])
    probabilities = raw_data.get('inference_results', {}).get('probabilities', [])
    tokens = raw_data.get('tokens', [])

    print(f"      Total tokens in file: {len(predictions)}")

    # Extraire uniquement les positions avec pred=1 (boundary)
    boundary_positions = []

    for i, pred in enumerate(predictions):
        if pred == 1:  # C'est un boundary token
            if i < len(offsets):
                offset = offsets[i]
                if offset != [0, 0]:  # Ignorer les tokens spéciaux
                    prob = probabilities[i] if i < len(probabilities) else 0
                    token_str = tokens[i] if i < len(tokens) else ""

                    boundary_positions.append({
                        'char_start': offset[0],
                        'char_end': offset[1],
                        'token': token_str,
                        'probability': prob,
                    })

    # Trier par position
    boundary_positions.sort(key=lambda x: x['char_start'])

    print(f"      Boundary tokens trouvés: {len(boundary_positions)}")

    return boundary_positions


def create_boundary_clusters(boundary_positions: List[Dict], gap_threshold: int = 50) -> List[Dict]:
    """
    Créer des clusters de boundary tokens consécutifs.

    Un cluster = groupe de boundary tokens séparés par moins de gap_threshold chars.
    Chaque cluster représente un isnad candidate détecté par CAMeL-BERT.
    """
    print(f"\n[3/5] Créer les clusters de boundary tokens (gap threshold: {gap_threshold} chars)...")

    if not boundary_positions:
        return []

    clusters = []
    current_cluster = {
        'cluster_start': boundary_positions[0]['char_start'],
        'cluster_end': boundary_positions[0]['char_end'],
        'tokens': [boundary_positions[0]],
        'token_count': 1,
    }

    for pos in boundary_positions[1:]:
        # Vérifier s'il y a un gap trop grand
        gap = pos['char_start'] - current_cluster['cluster_end']

        if gap <= gap_threshold:
            # Étendre le cluster
            current_cluster['cluster_end'] = max(current_cluster['cluster_end'], pos['char_end'])
            current_cluster['tokens'].append(pos)
            current_cluster['token_count'] += 1
        else:
            # Fermer le cluster et en commencer un nouveau
            clusters.append(current_cluster)
            current_cluster = {
                'cluster_start': pos['char_start'],
                'cluster_end': pos['char_end'],
                'tokens': [pos],
                'token_count': 1,
            }

    # Ne pas oublier le dernier cluster
    clusters.append(current_cluster)

    print(f"      Clusters créés: {len(clusters)}")
    print(f"      Tokens par cluster (moyenne): {sum(c['token_count'] for c in clusters) / len(clusters):.1f}")

    return clusters


def calculate_overlap(seg1_start: int, seg1_end: int, seg2_start: int, seg2_end: int) -> float:
    """Calculer le pourcentage de chevauchement entre deux segments."""
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap = overlap_end - overlap_start
    seg1_length = seg1_end - seg1_start

    return (overlap / seg1_length) * 100 if seg1_length > 0 else 0.0


def compare_isnads(isnads: List[Dict], clusters: List[Dict]) -> Dict:
    """
    Comparer les positions des isnads gold standard avec les clusters CAMeL-BERT.
    """
    print("\n[4/5] Comparer les isnads...")

    print("\n" + "="*80)
    print("COMPARAISON ISNADS: GOLD STANDARD vs CAMeL-BERT")
    print("="*80)

    print(f"\nGold Standard:")
    print(f"  Isnads trouvés: {len(isnads)}")
    print(f"  Position moyenne: [{sum(i['isnad_start'] for i in isnads)/len(isnads):.0f}]")

    print(f"\nCAMeL-BERT:")
    print(f"  Clusters détectés: {len(clusters)}")
    if clusters:
        print(f"  Position moyenne: [{sum(c['cluster_start'] for c in clusters)/len(clusters):.0f}]")

    # Analyser les overlaps
    print(f"\n" + "-"*80)
    print("ANALYSE DES OVERLAPS (Isnad Gold → Cluster CAMeL-BERT)")
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

            # Calculer la distance (début du cluster au début du isnad)
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
            'isnad_text_sample': isnad['isnad_text'],
        })

    # Statistiques
    perfect_matches = sum(1 for m in matches if m['best_overlap'] >= 90)
    good_matches = sum(1 for m in matches if 70 <= m['best_overlap'] < 90)
    acceptable_matches = sum(1 for m in matches if 50 <= m['best_overlap'] < 70)
    bad_matches = sum(1 for m in matches if m['best_overlap'] < 50)

    if len(matches) > 0:
        avg_overlap = sum(m['best_overlap'] for m in matches) / len(matches)
        median_distance = sorted([m['best_distance'] for m in matches if m['best_distance'] is not None])[len([m for m in matches if m['best_distance'] is not None]) // 2]
    else:
        avg_overlap = 0
        median_distance = 0

    print(f"\nChevauchement (Overlap):")
    print(f"  Moyen: {avg_overlap:.1f}%")
    print(f"  Parfaits (≥90%): {perfect_matches}/{len(matches)} ({100*perfect_matches/len(matches):.1f}%)")
    print(f"  Bons (70-90%): {good_matches}/{len(matches)}")
    print(f"  Acceptables (50-70%): {acceptable_matches}/{len(matches)}")
    print(f"  Mauvais (<50%): {bad_matches}/{len(matches)}")

    print(f"\nDistance du début (Cluster vs Isnad):")
    print(f"  Médiane: {median_distance:.0f} chars")

    # Exemples de bons matches
    print(f"\n" + "-"*80)
    print("EXEMPLES DE BONS MATCHES (≥90% overlap)")
    print("-"*80)

    good_examples = [m for m in matches if m['best_overlap'] >= 90][:5]
    for m in good_examples:
        print(f"\n  Akhbar #{m['akhbar_num']}:")
        print(f"    Gold:    [{m['gold_start']:6d}:{m['gold_end']:6d}]")
        if m['best_cluster_idx'] != -1:
            c = clusters[m['best_cluster_idx']]
            print(f"    Cluster: [{c['cluster_start']:6d}:{c['cluster_end']:6d}] (tokens: {c['token_count']})")
        print(f"    Overlap: {m['best_overlap']:.1f}%")
        print(f"    Isnad: {m['isnad_text_sample']}...")

    # Exemples de mauvais matches
    print(f"\n" + "-"*80)
    print("EXEMPLES DE MAUVAIS MATCHES (<50% overlap)")
    print("-"*80)

    bad_examples = [m for m in matches if m['best_overlap'] < 50][:5]
    if bad_examples:
        for m in bad_examples:
            print(f"\n  Akhbar #{m['akhbar_num']}:")
            print(f"    Gold:    [{m['gold_start']:6d}:{m['gold_end']:6d}]")
            if m['best_cluster_idx'] != -1:
                c = clusters[m['best_cluster_idx']]
                print(f"    Cluster: [{c['cluster_start']:6d}:{c['cluster_end']:6d}]")
            else:
                print(f"    Cluster: AUCUN CLUSTER DÉTECTÉ")
            print(f"    Overlap: {m['best_overlap']:.1f}%")
            print(f"    Isnad: {m['isnad_text_sample']}...")
    else:
        print("      Aucun mauvais match!")

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
    }


def save_results(comparison: Dict, isnads: List[Dict], clusters: List[Dict]):
    """Sauvegarder les résultats de la validation."""
    print("\n[5/5] Sauvegarder les résultats...")

    results = {
        'validation_date': '2026-04-22',
        'gold_standard': {
            'count': comparison['gold_count'],
            'isnads': isnads,
        },
        'camelbert': {
            'count': comparison['camelbert_count'],
            'clusters': clusters,
        },
        'comparison': {
            'perfect_matches': comparison['perfect_matches'],
            'good_matches': comparison['good_matches'],
            'acceptable_matches': comparison['acceptable_matches'],
            'bad_matches': comparison['bad_matches'],
            'avg_overlap': round(comparison['avg_overlap'], 2),
            'median_distance': round(comparison['median_distance'], 1),
            'matches': comparison['matches'],
        },
    }

    output_path = 'results/validation_camelbert_isnads.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✓ Résultats sauvegardés: {output_path}")

    # Résumé final
    print(f"\n" + "="*80)
    print("RÉSUMÉ FINAL")
    print("="*80)
    print(f"\nCAMeL-BERT détecte les isnads avec une couverture de:")
    perfect_pct = 100 * comparison['perfect_matches'] / comparison['gold_count'] if comparison['gold_count'] > 0 else 0
    print(f"  ✓ Parfaite (≥90%): {perfect_pct:.1f}%")
    print(f"  ~ Bonne (70-90%): {100*comparison['good_matches']/comparison['gold_count']:.1f}%")
    print(f"  ≈ Acceptable (50-70%): {100*comparison['acceptable_matches']/comparison['gold_count']:.1f}%")
    print(f"  ✗ Mauvaise (<50%): {100*comparison['bad_matches']/comparison['gold_count']:.1f}%")
    print(f"\nOverlap moyen: {comparison['avg_overlap']:.1f}%")
    print(f"Distance médiane du début: {comparison['median_distance']:.0f} chars")


def main():
    # Charger le corpus
    corpus = load_corpus()
    print(f"Corpus: {len(corpus):,} caractères")

    # Extraire les positions des isnads
    isnads, gold_data = extract_isnad_positions_from_gold_standard(corpus)

    # Charger les boundary tokens de CAMeL-BERT
    boundary_positions = load_camelbert_boundary_positions()

    # Créer les clusters
    clusters = create_boundary_clusters(boundary_positions, gap_threshold=50)

    # Comparer
    comparison = compare_isnads(isnads, clusters)

    # Sauvegarder
    save_results(comparison, isnads, clusters)


if __name__ == "__main__":
    main()
