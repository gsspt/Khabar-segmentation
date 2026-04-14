"""
Préparer le dataset pour le fine-tuning AraBERT.
Convertir les 612 akhbars annotés en format BIO (token classification).
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import pyarabic.araby as arb
from transformers import AutoTokenizer


def load_annotated_akhbars(json_path: str) -> List[Dict]:
    """Charger le fichier JSON annoté des akhbars."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['akhbar']


def extract_akhbar_text(akhbar: Dict) -> str:
    """Extraire le texte complet d'un akhbar (tous ses segments concaténés)."""
    segments = akhbar['content']['segments']
    texts = [seg['text'] for seg in segments]
    return ' '.join(texts)


def preprocess_arabic_text(text: str) -> str:
    """Normaliser le texte arabe (supprimer diacritiques, normaliser espaces)."""
    # Supprimer diacritiques
    text = arb.strip_diacritics(text)
    # Normaliser espaces multiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize_arabic_text(text: str, tokenizer) -> List[str]:
    """Tokenizer le texte arabe avec le tokenizer AraBERT."""
    # Tokenizer HF
    tokens = tokenizer.tokenize(text)
    return tokens


def create_bio_labels(text: str, tokenizer) -> Tuple[List[str], List[str]]:
    """
    Créer les labels BIO pour un akhbar.
    B-KHABAR: début d'akhbar
    I-KHABAR: intérieur d'akhbar
    O: non-akhbar (ne devrait pas arriver ici puisque tout est un akhbar)
    """
    tokens = tokenize_arabic_text(text, tokenizer)

    # Tous les tokens d'un akhbar annoté = KHABAR
    # Premier token = B-KHABAR, reste = I-KHABAR
    labels = ['B-KHABAR'] + ['I-KHABAR'] * (len(tokens) - 1)

    return tokens, labels


def prepare_training_examples(akhbars: List[Dict], tokenizer, max_length: int = 512) -> List[Dict]:
    """
    Préparer les exemples d'entraînement.
    Chaque exemple = (tokens, labels) pour un ou plusieurs akhbars concaténés.
    """
    examples = []

    for i, akhbar in enumerate(akhbars):
        try:
            # Extraire et normaliser le texte
            text = extract_akhbar_text(akhbar)
            text = preprocess_arabic_text(text)

            if not text:
                print(f"Warning: Akhbar {akhbar['num']} est vide, skipped")
                continue

            # Tokenizer et créer labels
            tokens, labels = create_bio_labels(text, tokenizer)

            # Découper si trop long
            if len(tokens) > max_length:
                # Découper en chunks avec overlap
                stride = max_length // 2
                for start in range(0, len(tokens), stride):
                    end = min(start + max_length, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunk_labels = labels[start:end]

                    # Marquer le début du chunk comme B-KHABAR si ce n'est pas le début initial
                    if start > 0 and len(chunk_labels) > 0:
                        chunk_labels[0] = 'B-KHABAR'

                    examples.append({
                        'tokens': chunk_tokens,
                        'labels': chunk_labels,
                        'akhbar_num': akhbar['num'],
                        'subtype': akhbar.get('subtype', 'unknown')
                    })

                    if end >= len(tokens):
                        break
            else:
                examples.append({
                    'tokens': tokens,
                    'labels': labels,
                    'akhbar_num': akhbar['num'],
                    'subtype': akhbar.get('subtype', 'unknown')
                })

        except Exception as e:
            print(f"Error processing akhbar {akhbar.get('num', i)}: {e}")
            continue

    return examples


def save_examples(examples: List[Dict], output_path: str):
    """Sauvegarder les exemples en format JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write('\n')
    print(f"[OK] Saved {len(examples)} examples to {output_path}")


def main():
    # Chemin des données
    json_path = Path(__file__).parent.parent / 'data' / 'processed' / 'Kitab_Uqala_al_Majanin_annotated.json'

    # Créer le dossier processed s'il n'existe pas
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Vérifier si le fichier annoté existe dans Uqala_NLP
    if not json_path.exists():
        json_path = Path(__file__).parent.parent.parent / 'Uqala_NLP' / 'data' / 'annotated' / 'Kitab_Uqala_al_Majanin_annotated.json'

    if not json_path.exists():
        raise FileNotFoundError(f"JSON annoté non trouvé: {json_path}")

    print(f"Loading annotated akhbars from {json_path}...")
    akhbars = load_annotated_akhbars(str(json_path))
    print(f"[OK] Loaded {len(akhbars)} akhbars")

    # Charger le tokenizer AraBERT
    print("Loading AraBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')

    # Préparer les exemples
    print("Preparing training examples...")
    examples = prepare_training_examples(akhbars, tokenizer)
    print(f"[OK] Generated {len(examples)} training examples")

    # Train/val split (simple split sans stratification pour éviter les problèmes de classes minoritaires)
    train_examples, val_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=42,
    )

    # Sauvegarder
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    save_examples(train_examples, str(output_dir / 'train_akhbars_bio.jsonl'))
    save_examples(val_examples, str(output_dir / 'val_akhbars_bio.jsonl'))

    print(f"\n[OK] Dataset prepared!")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")
    print(f"  Total tokens (approx): {sum(len(ex['tokens']) for ex in train_examples + val_examples)}")


if __name__ == '__main__':
    main()
