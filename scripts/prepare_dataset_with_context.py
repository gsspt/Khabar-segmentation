"""
Préparer dataset d'entraînement avec contexte O réel (avec nettoyage agressif OpenITI).
Utilise les alignements fuzzy pour extraire les akhbars du texte brut avec contexte.
Applique le même nettoyage que evaluate.py pour assurer la cohérence.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from tqdm import tqdm


def clean_openiti_text(text: str) -> str:
    """Nettoyer ULTRA-AGRESSIVEMENT: garder SEULEMENT du texte arabe.

    1. Supprime explicitement tous les # (métadonnées OpenITI)
    2. Garde SEULEMENT les caractères arabes et espaces
    """
    # 1. Supprimer TOUS les # (peu importe le contexte)
    text = text.replace('#', ' ')

    # 2. Supprimer TOUS les caractères non-arabes et non-espaces
    # Garder seulement:
    # - U+0600-U+06FF: Arabic
    # - U+0750-U+077F: Arabic Supplement
    # - U+FB50-U+FDFF: Arabic Presentation Forms-A
    # - U+FE70-U+FEFF: Arabic Presentation Forms-B
    # - Espaces et newlines
    text = re.sub(
        r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s]',
        ' ',
        text
    )

    # 3. Normaliser les espaces multiples et newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def load_raw_text(text_path: str) -> str:
    """Charger et nettoyer le texte brut OpenITI."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Appliquer le nettoyage agressif
    text = clean_openiti_text(text)
    return text


def load_alignment_results(alignment_path: str) -> Dict:
    """Charger les résultats d'alignement fuzzy."""
    with open(alignment_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_annotated_akhbars(json_path: str) -> Dict[int, str]:
    """Charger et nettoyer les textes des akhbars annotés."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    akhbars_dict = {}
    for akh in data['akhbar']:
        segments = akh['content']['segments']
        text = ' '.join([seg['text'] for seg in segments])
        # Appliquer le nettoyage aussi au texte d'akhbar (au cas où il contient des marqueurs résiduels)
        text = clean_openiti_text(text)
        akhbars_dict[akh['num']] = text

    return akhbars_dict


def extract_context_span(
    raw_text: str,
    position: int,
    akhbar_text: str,
    context_size: int = 300
) -> Tuple[str, int, int]:
    """
    Extraire le span d'akhbar + contexte du texte brut.

    Retourne: (texte_avec_contexte, position_akhbar_dans_span, longueur_akhbar)
    """
    # Estimer la longueur approximative de l'akhbar dans le texte brut
    # (après normalisation, peut être différent)
    approx_length = int(len(akhbar_text) * 1.2)

    # Bornes du span avec contexte
    start = max(0, position - context_size)
    end = min(len(raw_text), position + approx_length + context_size)

    span_text = raw_text[start:end]
    akhbar_pos_in_span = position - start
    akhbar_length = end - position  # Approximation

    return span_text, akhbar_pos_in_span, akhbar_length


def create_bio_labels_from_akhbar_text(
    tokens: List[str],
    full_span_text: str,
    akhbar_text: str,
    tokenizer
) -> List[str]:
    """
    Créer les labels BIO en cherchant le texte d'akhbar dans le span.
    Plus robuste que le mappage de positions.
    """
    # Normaliser pour la recherche
    import re

    def normalize_for_search(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\u064B-\u065F]', '', text)
        return text.strip()

    span_normalized = normalize_for_search(full_span_text)
    akhbar_normalized = normalize_for_search(akhbar_text)

    # Chercher l'akhbar dans le span normalisé
    akhbar_pos = span_normalized.find(akhbar_normalized)

    if akhbar_pos == -1:
        # Fallback: utiliser une approche heuristique
        # Assumer que l'akhbar commence après un certain point
        akhbar_pos = len(span_normalized) // 4  # Approximation

    akhbar_end_pos = akhbar_pos + len(akhbar_normalized)

    labels = []
    is_in_akhbar = False
    char_pos = 0

    # Construire un mapping token -> char position
    for token in tokens:
        clean_token = token.lstrip('#')

        # Vérifier si ce token est dans l'akhbar (approximativement)
        # En utilisant le ratio de position
        token_ratio_start = char_pos / max(len(span_normalized), 1)
        token_ratio_end = (char_pos + len(clean_token)) / max(len(span_normalized), 1)

        akhbar_ratio_start = akhbar_pos / max(len(span_normalized), 1)
        akhbar_ratio_end = akhbar_end_pos / max(len(span_normalized), 1)

        # Token chevauche l'akhbar?
        overlaps = (
            token_ratio_start < akhbar_ratio_end and
            token_ratio_end > akhbar_ratio_start
        )

        if overlaps:
            if not is_in_akhbar:
                # Début de l'akhbar
                labels.append('B-KHABAR')
                is_in_akhbar = True
            else:
                # Continuation de l'akhbar
                labels.append('I-KHABAR')
        else:
            # Token hors de l'akhbar
            labels.append('O')
            is_in_akhbar = False

        char_pos += len(clean_token) + 1  # +1 pour l'espace

    return labels


def prepare_training_examples(
    raw_text: str,
    alignment_results: Dict,
    akhbars_dict: Dict[int, str],
    tokenizer,
    min_score: float = 70.0,
    context_size: int = 300
) -> List[Dict]:
    """
    Préparer les exemples d'entraînement avec contexte réel.
    """
    examples = []

    print(f"[*] Filtering alignments with score >= {min_score}...")
    good_alignments = [
        a for a in alignment_results['alignment_map']
        if a['score'] >= min_score and a['found']
    ]
    print(f"[OK] {len(good_alignments)} good alignments")

    print(f"[*] Creating training examples with context...")

    for alignment in tqdm(good_alignments, desc="Processing alignments"):
        akh_num = alignment['akhbar_num']
        position = alignment['position']

        # Récupérer le texte d'akhbar
        if akh_num not in akhbars_dict:
            continue

        akhbar_text = akhbars_dict[akh_num]

        # Extraire le span avec contexte
        try:
            span_text, akh_start_in_span, akh_length = extract_context_span(
                raw_text, position, akhbar_text, context_size
            )
        except Exception as e:
            print(f"[!] Error extracting span for akhbar {akh_num}: {e}")
            continue

        if len(span_text.split()) < 5:
            continue  # Span trop court

        # Tokenizer le span
        try:
            tokens = tokenizer.tokenize(span_text)
        except Exception as e:
            print(f"[!] Error tokenizing akhbar {akh_num}: {e}")
            continue

        if len(tokens) == 0:
            continue

        # Estimer les positions caractères de l'akhbar dans le span
        # C'est approximatif, on utilise la position relative
        akh_start_char = akh_start_in_span
        akh_end_char = akh_start_in_span + akh_length

        # Créer les labels BIO
        try:
            labels = create_bio_labels_from_akhbar_text(
                tokens, span_text, akhbar_text, tokenizer
            )
        except Exception as e:
            print(f"[!] Error creating labels for akhbar {akh_num}: {e}")
            continue

        # Vérifier que tokens et labels ont la même longueur
        if len(tokens) != len(labels):
            # Ajuster si nécessaire
            min_len = min(len(tokens), len(labels))
            tokens = tokens[:min_len]
            labels = labels[:min_len]

        if len(tokens) < 2:
            continue

        examples.append({
            'tokens': tokens,
            'labels': labels,
            'akhbar_num': akh_num,
            'length': len(tokens),
            'alignment_score': alignment['score'],
        })

    return examples


def save_examples(examples: List[Dict], output_path: str):
    """Sauvegarder les exemples en JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            json.dump(ex, f, ensure_ascii=False)
            f.write('\n')
    print(f"[OK] Saved {len(examples)} examples to {output_path}")


def main():
    # Chemins
    raw_text_path = Path('../Uqala_NLP/openiti_corpus/data/0406IbnHabibNaysaburi/0406IbnHabibNaysaburi.CuqalaMajanin/0406IbnHabibNaysaburi.CuqalaMajanin.Shamela0001593-ara1')
    alignment_path = Path('./results/fuzzy_alignment_advanced.json')
    json_path = Path('./data/processed/Kitab_Uqala_al_Majanin_annotated.json')
    output_dir = Path('./data/processed')

    # Vérifier les fichiers
    if not raw_text_path.exists():
        print(f"[!] Raw text not found: {raw_text_path}")
        return

    if not alignment_path.exists():
        print(f"[!] Alignment results not found: {alignment_path}")
        print("    Run fuzzy_alignment_advanced.py first")
        return

    if not json_path.exists():
        print(f"[!] JSON not found: {json_path}")
        return

    # Charger les données
    print("[*] Loading raw text...")
    raw_text = load_raw_text(str(raw_text_path))

    print("[*] Loading alignment results...")
    alignment_results = load_alignment_results(str(alignment_path))

    print("[*] Loading annotated akhbars...")
    akhbars_dict = load_annotated_akhbars(str(json_path))

    print("[*] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')

    # Préparer les exemples
    print()
    examples = prepare_training_examples(
        raw_text, alignment_results, akhbars_dict, tokenizer,
        min_score=70.0,
        context_size=300
    )

    print(f"[OK] Generated {len(examples)} training examples with context")

    # Statistiques
    lengths = [ex['length'] for ex in examples]
    scores = [ex['alignment_score'] for ex in examples]

    print(f"\n[STATISTICS]")
    print(f"  Min length: {min(lengths)} tokens")
    print(f"  Max length: {max(lengths)} tokens")
    print(f"  Mean length: {sum(lengths) / len(lengths):.0f} tokens")
    print(f"  Average alignment score: {sum(scores) / len(scores):.2f}")

    # Sauvegarder
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'train_akhbars_with_context.jsonl'
    save_examples(examples, str(output_path))

    print(f"\n[OK] Dataset prepared with context!")
    print(f"    Output: {output_path}")
    print(f"\nNext step:")
    print(f"    python scripts/train_with_camelbert.py")


if __name__ == '__main__':
    main()
