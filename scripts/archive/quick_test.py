"""
Quick test : tester le modele sur des akhbars complets du corpus annoté.
Version corrigée pour Colab - utilise entity_group au lieu de entity.
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np


def quick_test(model_path=None):
    """Tester le modele sur des akhbars complets."""

    # Chemin du modèle (par défaut pour Colab)
    if model_path is None:
        possible_paths = [
            Path('/content/drive/MyDrive/Thèse/khabar-segmentation/checkpoints/arabertv2_akhbars_v2'),
            Path('./checkpoints/arabertv2_akhbars_v2'),
            Path('./checkpoints/arabertv2_akhbars'),
        ]
    else:
        possible_paths = [Path(model_path)]

    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        print("[!] Model checkpoint not found!")
        print(f"    Tried: {[str(p) for p in possible_paths]}")
        return

    print(f"[*] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    print(f"[OK] Model loaded!")
    print(f"     Labels: {model.config.id2label}\n")

    # Pipeline pour token classification
    nlp = pipeline(
        'token-classification',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        aggregation_strategy='simple'
    )

    # Charger les akhbars du fichier annoté
    print("[*] Loading test akhbars from annotated JSON...")
    json_path = Path('./data/processed/Kitab_Uqala_al_Majanin_annotated.json')

    if not json_path.exists():
        print(f"[!] JSON file not found: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prendre les 5 premiers akhbars complets
    test_akhbars = []
    for akh in data['akhbar'][:5]:
        # Combiner tous les segments
        text = ' '.join([seg['text'] for seg in akh['content']['segments']])
        test_akhbars.append({
            'num': akh['num'],
            'text': text,
            'subtype': akh.get('subtype', 'unknown'),
            'length': len(text.split())
        })

    print(f"[OK] Loaded {len(test_akhbars)} akhbars for testing\n")

    print("[*] Testing on full akhbars...")
    print("=" * 80)

    stats = {
        'total': len(test_akhbars),
        'recognized': 0,
        'avg_confidence': 0.0,
        'confidences': []
    }

    for akh_info in test_akhbars:
        print(f"\n[AKHBAR {akh_info['num']}] (subtype: {akh_info['subtype']}, length: {akh_info['length']} words)")
        print(f"\n--- FULL TEXT ---")
        print(akh_info['text'])
        print("--- END TEXT ---\n")

        try:
            sentence = akh_info['text']

            # Diviser en chunks si trop long
            if len(sentence) > 2000:
                sentence = sentence[:2000]
                print(f"  (input truncated to 2000 chars for pipeline)")

            predictions = nlp(sentence)

            # Extraire les predictions (nouvelle structure avec entity_group)
            khabar_tokens = []
            khabar_scores = []

            for pred in predictions:
                # Utiliser entity_group au lieu de entity
                entity_group = pred.get('entity_group', 'O')
                score = pred.get('score', 0.0)
                word = pred.get('word', '')

                if entity_group == 'KHABAR':
                    khabar_tokens.append(word)
                    khabar_scores.append(score)

            if khabar_tokens:
                avg_score = np.mean(khabar_scores) if khabar_scores else 0.0
                stats['recognized'] += 1
                stats['confidences'].append(avg_score)

                print(f"[SUCCESS] Recognized as KHABAR: YES")
                print(f"  Confidence: {avg_score:.4f}")
                print(f"  Tokens recognized: {len(khabar_tokens)}")
                print(f"\n--- EXTRACTED TEXT ---")
                print(' '.join(khabar_tokens))
                print("--- END EXTRACTED ---\n")
            else:
                print(f"[FAIL] Recognized as KHABAR: NO")
                print(f"  (Model didn't recognize this as a khabar)\n")

        except Exception as e:
            print(f"  Error: {e}\n")

    print("=" * 80)
    print(f"\n[RESULTS]")
    print(f"  Total tested: {stats['total']}")
    print(f"  Recognized: {stats['recognized']}/{stats['total']}")
    if stats['confidences']:
        print(f"  Average confidence: {np.mean(stats['confidences']):.4f}")
    print(f"\n[OK] Quick test complete!")


if __name__ == '__main__':
    quick_test()
