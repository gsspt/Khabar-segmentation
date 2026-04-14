"""
Préparer le dataset pour le fine-tuning AraBERT - Version 2.
Sans découpage : chaque akhbar reste intact, peu importe sa longueur.
"""

import json
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm


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


def preprocess_text(text: str) -> str:
    """Normaliser les espaces multiples."""
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def prepare_training_examples(akhbars: List[Dict], tokenizer) -> List[Dict]:
    """
    Préparer les exemples d'entraînement.
    IMPORTANT : Chaque akhbar reste ENTIER, pas de découpage.
    """
    examples = []

    print("[*] Preparing training examples (NO SPLITTING)...")

    for akhbar in tqdm(akhbars, desc="Processing akhbars"):
        try:
            # Extraire et normaliser le texte
            text = extract_akhbar_text(akhbar)
            text = preprocess_text(text)

            if not text:
                continue

            # Tokenizer
            tokens = tokenizer.tokenize(text)

            if len(tokens) == 0:
                continue

            # Labels : tous les tokens = KHABAR
            # Premier token = B-KHABAR, reste = I-KHABAR
            labels = ['B-KHABAR'] + ['I-KHABAR'] * (len(tokens) - 1)

            examples.append({
                'tokens': tokens,
                'labels': labels,
                'akhbar_num': akhbar['num'],
                'length': len(tokens),
            })

        except Exception as e:
            print(f"[!] Error processing akhbar {akhbar.get('num')}: {e}")
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
    # Chemins
    json_path = Path('./data/processed/Kitab_Uqala_al_Majanin_annotated.json')

    if not json_path.exists():
        raise FileNotFoundError(f"JSON non trouvé: {json_path}")

    print(f"[*] Loading annotated akhbars from {json_path}...")
    akhbars = load_annotated_akhbars(str(json_path))
    print(f"[OK] Loaded {len(akhbars)} akhbars")

    # Charger le tokenizer
    print("[*] Loading AraBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')

    # Préparer les exemples (sans découpage)
    examples = prepare_training_examples(akhbars, tokenizer)
    print(f"[OK] Generated {len(examples)} training examples")

    # Statistiques
    lengths = [ex['length'] for ex in examples]
    print(f"\n[STATISTICS]")
    print(f"  Min length: {min(lengths)} tokens")
    print(f"  Max length: {max(lengths)} tokens")
    print(f"  Mean length: {sum(lengths) / len(lengths):.0f} tokens")

    # Train/val split
    train_examples, val_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=42,
    )

    # Sauvegarder
    output_dir = Path('./data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)

    save_examples(train_examples, str(output_dir / 'train_akhbars_bio.jsonl'))
    save_examples(val_examples, str(output_dir / 'val_akhbars_bio.jsonl'))

    print(f"\n[OK] Dataset prepared!")
    print(f"  Train: {len(train_examples)} examples")
    print(f"  Val: {len(val_examples)} examples")


if __name__ == '__main__':
    main()
