#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison CAMeL-BERT vs Gold Standard
SANS PyTorch - utilise seulement le JSON et des calculs simples
"""

import json
from pathlib import Path


def load_data():
    """Charger tous les fichiers nécessaires."""
    print("[1/3] Charger les données...")

    # Corpus
    with open('data/processed/kitab_uqala_reference_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      Corpus: {len(text):,} caractères")

    # Gold standard
    with open('data/processed/kitab_uqala_boundaries.json', 'r', encoding='utf-8') as f:
        gold_standard = json.load(f)
    print(f"      Gold Standard: {len(gold_standard)} boundaries")

    # CAMeL-BERT results
    with open('results/camelbert_boundary_tokens_clean.json', 'r', encoding='utf-8') as f:
        camelbert_raw = json.load(f)
    print(f"      CAMeL-BERT: {len(camelbert_raw['boundary_tokens']):,} boundary tokens")

    return text, gold_standard, camelbert_raw


def estimate_camelbert_segments(text, boundary_tokens, boundary_indices):
    """
    Estimer les segments CAMeL-BERT EN CHERCHANT les tokens dans le texte.
    Note: c'est une approximation sans le tokenizer exact.
    """
    print("\n[2/3] Estimer les segments CAMeL-BERT...")

    segments = []

    # Chercher les positions approximatives des boundary tokens
    boundary_positions = []

    for token in set(boundary_tokens):
        # Chercher toutes les occurrences du token dans le texte
        pos = 0
        while True:
            idx = text.find(token, pos)
            if idx == -1:
                break
            boundary_positions.append(idx)
            pos = idx + 1

    # Trier et créer des segments approximatifs
    boundary_positions = sorted(set(boundary_positions))

    for i in range(len(boundary_positions) - 1):
        segments.append({
            'start': boundary_positions[i],
            'end': boundary_positions[i+1],
            'length': boundary_positions[i+1] - boundary_positions[i]
        })

    # Dernier segment
    if boundary_positions:
        segments.append({
            'start': boundary_positions[-1],
            'end': len(text),
            'length': len(text) - boundary_positions[-1]
        })

    print(f"      {len(segments)} segments estimés")
    return segments


def calculate_overlap(seg1_start, seg1_end, seg2_start, seg2_end):
    """Calculer le pourcentage de chevauchement."""
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap = overlap_end - overlap_start
    seg1_length = seg1_end - seg1_start

    return (overlap / seg1_length) * 100 if seg1_length > 0 else 0.0


def compare(text, gold_standard, camelbert_segments):
    """Comparer les deux approches."""
    print("\n[3/3] Analyser les résultats...")

    print("\n" + "="*80)
    print("COMPARAISON CAMeL-BERT vs GOLD STANDARD")
    print("="*80)

    print(f"\nGold Standard:")
    print(f"  Boundaries: {len(gold_standard)}")
    total_gold_chars = sum(b['end'] - b['start'] for b in gold_standard)
    print(f"  Caractères couverts: {total_gold_chars:,}")
    print(f"  Longueur moyenne: {total_gold_chars/len(gold_standard):.0f} chars")

    print(f"\nCAMeL-BERT:")
    print(f"  Segments estimés: {len(camelbert_segments)}")
    total_camelbert_chars = sum(s['length'] for s in camelbert_segments)
    print(f"  Caractères couverts: {total_camelbert_chars:,}")
    print(f"  Longueur moyenne: {total_camelbert_chars/len(camelbert_segments):.0f} chars")

    # Analyser les overlaps
    print(f"\n" + "-"*80)
    print("ANALYSE DES OVERLAPS")
    print("-"*80)

    overlaps = []
    for camelbert_seg in camelbert_segments:
        best_overlap = 0
        best_gold_idx = -1

        for gold_idx, gold_seg in enumerate(gold_standard):
            overlap = calculate_overlap(
                camelbert_seg['start'], camelbert_seg['end'],
                gold_seg['start'], gold_seg['end']
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_gold_idx = gold_idx

        overlaps.append({
            'camelbert_idx': len(overlaps),
            'overlap': best_overlap,
            'gold_match': best_gold_idx
        })

    perfect = sum(1 for o in overlaps if o['overlap'] >= 90)
    good = sum(1 for o in overlaps if 70 <= o['overlap'] < 90)
    bad = sum(1 for o in overlaps if o['overlap'] < 50)
    avg_overlap = sum(o['overlap'] for o in overlaps) / len(overlaps)

    print(f"\nChevauchement CAMeL-BERT → Gold:")
    print(f"  Moyen: {avg_overlap:.1f}%")
    print(f"  Parfaits (≥90%): {perfect}/{len(overlaps)} ({100*perfect/len(overlaps):.1f}%)")
    print(f"  Bons (70-90%): {good}")
    print(f"  Mauvais (<50%): {bad}")

    # Résumé
    print(f"\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if avg_overlap > 85:
        print("✓ CAMeL-BERT aligne BIEN avec le gold standard!")
    elif avg_overlap > 70:
        print("~ CAMeL-BERT aligne MOYENNEMENT avec le gold standard")
    else:
        print("✗ CAMeL-BERT aligne MAL avec le gold standard")

    return {
        'gold_count': len(gold_standard),
        'camelbert_count': len(camelbert_segments),
        'perfect_overlaps': perfect,
        'avg_overlap': avg_overlap,
    }


def main():
    text, gold_standard, camelbert_raw = load_data()

    camelbert_segments = estimate_camelbert_segments(
        text,
        camelbert_raw['boundary_tokens'],
        camelbert_raw['boundary_indices']
    )

    results = compare(text, gold_standard, camelbert_segments)

    # Sauvegarder
    with open('results/comparison_simple.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Résultats sauvegardés: results/comparison_simple.json")


if __name__ == "__main__":
    main()
