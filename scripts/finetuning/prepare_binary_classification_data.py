#!/usr/bin/env python3
"""
Prépare les données pour Piste 1 : Binary Classification (Boundary Detection)

Stratégie :
1. Charger Kitab_Uqala_al_Majanin_annotated.json
2. Pour chaque akhbar:
   - Concaténer TOUS les segments (isnad + matn + poetry + prose)
   - Étiqueter: segment.type == 'isnad' → label 1 (boundary), sinon 0
3. Tokenizer CAMeL-BERT pour créer tokens + labels
4. Nettoyer les isnads (supprimer numérals arabes + romains)
5. Exporter en format HuggingFace Datasets (train/val/test)

Format final:
{
  "akhbar_id": 2,
  "tokens": ["حدثنا", "محمد", "قال", "رأيت"],
  "labels": [1, 1, 1, 0],
  "segment_boundaries": [0, 3, 4]  # indices where segment.type changes
}
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Tenter import transformers
try:
    from transformers import AutoTokenizer
except ImportError:
    print("[WARNING] transformers not installed. Install with: pip install transformers")
    sys.exit(1)


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


# ── TOKENIZATION ET ÉTIQUETAGE ─────────────────────────────────────────────────

def prepare_akhbar_data(
    akhbar: Dict,
    tokenizer,
    model_name: str = "bert-base-multilingual-cased"
) -> Dict | None:
    """
    Prépare un akhbar pour la classification binaire.

    Returns:
        Dict avec clés: akhbar_id, tokens, labels, segment_types, isnad_indices
        ou None si l'akhbar est invalide
    """
    akhbar_id = akhbar.get('num')
    segments = akhbar['content']['segments']

    if not segments:
        return None

    # Construire la liste complète : (texte, type, texte_nettoyé)
    full_text_parts = []
    segment_info = []  # Trace des limites de segments

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
            full_text_parts.append(seg_text)
            segment_info.append({
                'type': seg_type,
                'text': seg_text,
                'start_idx': len(' '.join(full_text_parts).split()) if full_text_parts else 0
            })

    if not full_text_parts:
        return None

    full_text = ' '.join(full_text_parts)

    # Tokenizer
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False,  # Garder le texte complet
        padding=False
    )

    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])

    # Créer les labels
    # Stratégie : mapper character offsets → segment type
    labels = []
    char_to_segment_type = {}

    # Construire mapping caractère → type de segment
    char_pos = 0
    for part, seg_info_item in zip(full_text_parts, segment_info):
        for i in range(len(part)):
            char_to_segment_type[char_pos + i] = seg_info_item['type']
        char_pos += len(part) + 1  # +1 pour l'espace

    # Assigner les labels basé sur offsets
    for offset_start, offset_end in encoded['offset_mapping']:
        # Vérifier le type de segment au début du token
        if offset_start in char_to_segment_type:
            seg_type = char_to_segment_type[offset_start]
        else:
            # Chercher le type le plus proche
            seg_type = None
            for pos in range(offset_start, max(0, offset_start - 10), -1):
                if pos in char_to_segment_type:
                    seg_type = char_to_segment_type[pos]
                    break

        # Label : 1 si isnad, 0 sinon
        label = 1 if seg_type == 'isnad' else 0
        labels.append(label)

    # Vérifier que tokens et labels ont la même longueur
    assert len(tokens) == len(labels), \
        f"Mismatch: {len(tokens)} tokens vs {len(labels)} labels"

    return {
        'akhbar_id': akhbar_id,
        'tokens': tokens,
        'labels': labels,
        'segment_types': [s['type'] for s in segment_info],
        'has_isnad': any(s['type'] == 'isnad' for s in segment_info),
        'text': full_text,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    input_file = Path('data/processed/Kitab_Uqala_al_Majanin_annotated.json')
    output_dir = Path('data/processed/binary_classification_dataset')
    output_dir.mkdir(exist_ok=True)

    print(f"📖 Chargement du corpus... ({input_file})")
    data = json.loads(input_file.read_text(encoding='utf-8'))

    # Charger tokenizer CAMeL-BERT
    print("🔤 Chargement du tokenizer CAMeL-BERT...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-msa')
    except Exception as e:
        print(f"❌ Erreur lors du chargement du tokenizer: {e}")
        print("   Utilisation d'un tokenizer multilingual fallback...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Préparer les données
    print(f"\n📝 Traitement de {len(data['akhbar'])} akhbars...")

    valid_examples = []
    stats = defaultdict(int)

    for i, akhbar in enumerate(data['akhbar']):
        if i % 100 == 0:
            print(f"   [{i}/{len(data['akhbar'])}]")

        try:
            example = prepare_akhbar_data(akhbar, tokenizer)
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
            print(f"⚠️  Akhbar {akhbar.get('num')}: {e}")
            stats['error'] += 1

    print(f"\n✅ Statistiques:")
    print(f"   - Exemples valides: {stats['valid']}")
    print(f"   - Avec isnad: {stats['with_isnad']}")
    print(f"   - Sans isnad: {stats['without_isnad']}")
    print(f"   - Vides: {stats['empty']}")
    print(f"   - Erreurs: {stats['error']}")

    # Statistiques sur les labels
    total_tokens = sum(len(ex['tokens']) for ex in valid_examples)
    total_isnad_tokens = sum(sum(ex['labels']) for ex in valid_examples)
    boundary_ratio = total_isnad_tokens / total_tokens if total_tokens > 0 else 0

    print(f"\n📊 Statistiques des tokens:")
    print(f"   - Total tokens: {total_tokens}")
    print(f"   - Tokens boundary (isnad): {total_isnad_tokens}")
    print(f"   - Tokens non-boundary (matn/etc): {total_tokens - total_isnad_tokens}")
    print(f"   - Ratio boundary: {boundary_ratio:.2%}")

    # Sauvegarder les exemples
    output_file = output_dir / 'binary_classification_examples.jsonl'
    print(f"\n💾 Sauvegarde dans {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Sauvegarder les stats
    stats_file = output_dir / 'preparation_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_akhbars': len(data['akhbar']),
            'valid_examples': stats['valid'],
            'with_isnad': stats['with_isnad'],
            'without_isnad': stats['without_isnad'],
            'empty': stats['empty'],
            'errors': stats['error'],
            'total_tokens': total_tokens,
            'boundary_tokens': total_isnad_tokens,
            'non_boundary_tokens': total_tokens - total_isnad_tokens,
            'boundary_ratio': boundary_ratio,
        }, f, indent=2, ensure_ascii=False)

    print(f"✨ Préparation terminée!")
    print(f"   📄 Exemples: {output_file}")
    print(f"   📊 Stats: {stats_file}")

if __name__ == '__main__':
    main()
