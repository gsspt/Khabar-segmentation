#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparer CAMeL-BERT avec le gold standard
Convertit les résultats tokens → positions caractères
"""

import json
from pathlib import Path
from transformers import AutoTokenizer


def convert_token_indices_to_char_positions(
    corpus_path: str,
    boundary_tokens_json: str,
    model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
) -> dict:
    """
    Convertit les indices de tokens → positions caractères
    en utilisant les offsets du tokenizer.
    """

    print("[1/4] Charger le corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      {len(text):,} caractères")

    print("\n[2/4] Charger le tokenizer et tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokeniser avec offsets
    encoded = tokenizer(
        text,
        return_tensors='pt',
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    )

    token_ids = encoded['input_ids'][0].numpy()
    offsets = encoded['offset_mapping'][0].numpy()

    print(f"      {len(token_ids):,} tokens générés")

    print("\n[3/4] Charger les boundary tokens...")
    with open(boundary_tokens_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    boundary_indices = set(data['boundary_indices'])
    print(f"      {len(boundary_indices):,} boundary token indices")

    print("\n[4/4] Convertir en positions caractères...")

    # Créer les segments basés sur les tokens boundary
    # Un segment isnad commence quand on voit un boundary token
    # et se termine au prochain boundary token

    camelbert_segments = []
    current_segment_start = None
    current_segment_tokens = []

    for token_idx, (char_start, char_end) in enumerate(offsets):
        if token_idx in boundary_indices:
            # C'est un boundary token
            if current_segment_start is not None:
                # Fermer le segment précédent
                prev_segment_end = offsets[token_idx - 1][1]
                camelbert_segments.append({
                    'start': current_segment_start,
                    'end': prev_segment_end,
                    'type': 'isnad',
                    'token_count': len(current_segment_tokens)
                })
            # Commencer un nouveau segment
            current_segment_start = char_start
            current_segment_tokens = [token_idx]
        else:
            # Token non-boundary (prose)
            if current_segment_start is not None:
                current_segment_tokens.append(token_idx)

    # Fermer le dernier segment
    if current_segment_start is not None:
        camelbert_segments.append({
            'start': current_segment_start,
            'end': offsets[len(offsets) - 1][1],
            'type': 'isnad',
            'token_count': len(current_segment_tokens)
        })

    print(f"      {len(camelbert_segments)} segments créés")

    return {
        'text': text,
        'segments': camelbert_segments,
        'total_tokens': len(token_ids),
        'boundary_token_count': len(boundary_indices),
    }


def load_gold_standard(gold_standard_path: str) -> list:
    """Charger le gold standard."""
    with open(gold_standard_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_overlap(seg1_start, seg1_end, seg2_start, seg2_end) -> float:
    """Calculer le pourcentage de chevauchement entre deux segments."""
    overlap_start = max(seg1_start, seg2_start)
    overlap_end = min(seg1_end, seg2_end)

    if overlap_start >= overlap_end:
        return 0.0

    overlap = overlap_end - overlap_start
    seg1_length = seg1_end - seg1_start

    return (overlap / seg1_length) * 100 if seg1_length > 0 else 0.0


def compare_with_gold_standard(camelbert_data: dict, gold_standard: list) -> dict:
    """
    Comparer les résultats CAMeL-BERT avec le gold standard.
    """

    camelbert_segments = camelbert_data['segments']
    text = camelbert_data['text']

    print("\n" + "="*80)
    print("COMPARAISON CAMeL-BERT vs GOLD STANDARD")
    print("="*80)

    print(f"\nCAMeL-BERT:")
    print(f"  Segments: {len(camelbert_segments)}")
    print(f"  Tokens totaux: {camelbert_data['total_tokens']:,}")
    print(f"  Boundary tokens: {camelbert_data['boundary_token_count']:,}")

    print(f"\nGold Standard:")
    print(f"  Boundaries: {len(gold_standard)}")
    print(f"  Caractères: {sum(b['end'] - b['start'] for b in gold_standard):,}")

    # Analyser les overlaps
    print(f"\n" + "-"*80)
    print("ANALYSE DES OVERLAPS")
    print("-"*80)

    matches = []
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

        matches.append({
            'camelbert_idx': len(matches),
            'camelbert_start': camelbert_seg['start'],
            'camelbert_end': camelbert_seg['end'],
            'camelbert_length': camelbert_seg['end'] - camelbert_seg['start'],
            'best_gold_match_idx': best_gold_idx,
            'overlap_percentage': best_overlap,
        })

    # Statistiques
    perfect_matches = sum(1 for m in matches if m['overlap_percentage'] >= 90)
    avg_overlap = sum(m['overlap_percentage'] for m in matches) / len(matches)

    print(f"\nChevauchement CAMeL-BERT vs Gold:")
    print(f"  Moyen: {avg_overlap:.2f}%")
    print(f"  Parfaits (≥90%): {perfect_matches}/{len(matches)}")
    print(f"  Mauvais (<50%): {sum(1 for m in matches if m['overlap_percentage'] < 50)}")

    # Exemples
    print(f"\nExemples de segments:")
    for i in range(min(5, len(matches))):
        m = matches[i]
        text_sample = text[m['camelbert_start']:m['camelbert_start']+50].replace('\n', ' ')
        print(f"\n  CAMeL-BERT {i}:")
        print(f"    Position: [{m['camelbert_start']:6d}:{m['camelbert_end']:6d}]")
        print(f"    Texte: {text_sample}...")
        print(f"    Gold match: {m['best_gold_match_idx']} (overlap: {m['overlap_percentage']:.1f}%)")

    # Segment CAMeL-BERT sans bon match
    bad_matches = [m for m in matches if m['overlap_percentage'] < 50]
    if bad_matches:
        print(f"\nSegments CAMeL-BERT sans bon match au gold standard ({len(bad_matches)}):")
        for m in bad_matches[:3]:
            text_sample = text[m['camelbert_start']:m['camelbert_start']+50].replace('\n', ' ')
            print(f"  [{m['camelbert_start']:6d}:{m['camelbert_end']:6d}] {text_sample}...")

    return {
        'matches': matches,
        'perfect_matches': perfect_matches,
        'avg_overlap': avg_overlap,
        'camelbert_count': len(camelbert_segments),
        'gold_standard_count': len(gold_standard),
    }


def main():
    corpus_path = "data/processed/kitab_uqala_reference_corpus.txt"
    boundary_tokens_json = "results/camelbert_boundary_tokens_clean.json"
    gold_standard_path = "data/processed/kitab_uqala_boundaries.json"

    # Convertir tokens → positions caractères
    print("ÉTAPE 1: CONVERTIR LES RÉSULTATS TOKENS → CARACTÈRES")
    print("="*80)
    camelbert_data = convert_token_indices_to_char_positions(
        corpus_path,
        boundary_tokens_json
    )

    # Charger gold standard
    gold_standard = load_gold_standard(gold_standard_path)

    # Comparer
    print("\n\nÉTAPE 2: COMPARAISON")
    comparison = compare_with_gold_standard(camelbert_data, gold_standard)

    # Sauvegarder les résultats
    results = {
        'camelbert': {
            'segments': camelbert_data['segments'],
            'total_tokens': camelbert_data['total_tokens'],
            'boundary_tokens': camelbert_data['boundary_token_count'],
        },
        'gold_standard': gold_standard,
        'comparison': comparison,
    }

    output_path = 'results/comparison_camelbert_vs_gold.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n✓ Résultats sauvegardés: {output_path}")


if __name__ == "__main__":
    main()
