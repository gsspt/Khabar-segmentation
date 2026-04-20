#!/usr/bin/env python3
"""
Prépare les données pour Piste 1 : Binary Classification (sans dépendre de transformers)

VERSION ALLÉGÉE :
- Charger corpus + nettoyer isnads
- Créer JSON avec structure simple
- Tokenization CAMeL-BERT faite ultérieurement lors du fine-tuning

Cela évite les dépendances lourdes et permet une inspection rapide des données.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict


# ── NETTOYAGE DES ISNADS ──────────────────────────────────────────────────────

def remove_numerals(text: str) -> str:
    """Supprime numérals arabes et romains d'un isnad."""
    # Numérals arabes : ٠-٩
    text = re.sub(r'[٠-٩]', '', text)
    # Numérals romains : I, V, X, L, C, D, M (et variantes)
    text = re.sub(r'[IVXLCDM]+', '', text)
    # Chiffres latins simples : 0-9
    text = re.sub(r'[0-9]', '', text)
    return text


def normalize_text(text: str) -> str:
    """Normalise l'espace blanc."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_isnad(text: str) -> str:
    """Nettoie complètement un isnad."""
    text = remove_numerals(text)
    text = normalize_text(text)
    return text


# ── PRÉPARATION DES DONNÉES ───────────────────────────────────────────────────

def prepare_akhbar_data(akhbar: Dict) -> Dict | None:
    """
    Prépare un akhbar pour la classification binaire.

    Format final (sans tokenization, juste les segments bruts):
    {
        "akhbar_id": 2,
        "segments": [
            {"type": "isnad", "text": "حدثنا محمد"},
            {"type": "matn", "text": "قال رأيت"},
            ...
        ],
        "has_isnad": True
    }
    """
    akhbar_id = akhbar.get('num')
    segments = akhbar['content']['segments']

    if not segments:
        return None

    processed_segments = []

    for seg in segments:
        seg_type = seg.get('type', 'unknown')
        seg_text = seg.get('text', '').strip()

        if not seg_text:
            continue

        # Nettoyer les isnads
        if seg_type == 'isnad':
            seg_text = clean_isnad(seg_text)
        else:
            seg_text = normalize_text(seg_text)

        if seg_text:  # Ne pas ajouter les textes vides après nettoyage
            processed_segments.append({
                'type': seg_type,
                'text': seg_text,
                'length': len(seg_text)
            })

    if not processed_segments:
        return None

    return {
        'akhbar_id': akhbar_id,
        'segments': processed_segments,
        'has_isnad': any(s['type'] == 'isnad' for s in processed_segments),
        'segment_types': [s['type'] for s in processed_segments],
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    input_file = Path('data/processed/Kitab_Uqala_al_Majanin_annotated.json')
    output_dir = Path('data/processed/binary_classification_dataset')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading corpus...")
    data = json.loads(input_file.read_text(encoding='utf-8'))

    # Préparer les données
    print(f"[INFO] Processing {len(data['akhbar'])} akhbars...")

    valid_examples = []
    stats = defaultdict(int)

    for i, akhbar in enumerate(data['akhbar']):
        if i % 100 == 0:
            print(f"   [{i}/{len(data['akhbar'])}]")

        try:
            example = prepare_akhbar_data(akhbar)
            if example:
                valid_examples.append(example)
                stats['valid'] += 1
                if example['has_isnad']:
                    stats['with_isnad'] += 1
                else:
                    stats['without_isnad'] += 1
            else:
                stats['empty'] += 1
        except Exception as e:
            print(f"[ERROR] Akhbar {akhbar.get('num')}: {e}")
            stats['error'] += 1

    # Statistiques
    print(f"\n[STATS]")
    print(f"  Valid examples: {stats['valid']}")
    print(f"  With isnad: {stats['with_isnad']}")
    print(f"  Without isnad: {stats['without_isnad']}")
    print(f"  Empty: {stats['empty']}")
    print(f"  Errors: {stats['error']}")

    # Segments stats
    segment_type_counts = defaultdict(int)
    for ex in valid_examples:
        for seg in ex['segments']:
            segment_type_counts[seg['type']] += 1

    print(f"\n[SEGMENT TYPES]")
    for stype, count in sorted(segment_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {stype}: {count}")

    # Sauvegarder les exemples
    output_file = output_dir / 'binary_classification_examples.jsonl'
    print(f"\n[SAVE] Saving to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Sauvegarder stats
    stats_file = output_dir / 'preparation_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_akhbars': len(data['akhbar']),
            'valid_examples': stats['valid'],
            'with_isnad': stats['with_isnad'],
            'without_isnad': stats['without_isnad'],
            'empty': stats['empty'],
            'errors': stats['error'],
            'segment_type_counts': dict(segment_type_counts),
        }, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Done!")
    print(f"  Examples: {output_file}")
    print(f"  Stats: {stats_file}")
    print(f"\nNext step: run fine-tuning script with these prepared examples")

if __name__ == '__main__':
    main()
