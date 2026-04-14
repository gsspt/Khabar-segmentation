"""
Evaluer le modele fine-tune sur la version brute du corpus OpenITI.
Extraire les akhbars et comparer avec les annotations de reference.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np


LABEL2ID = {
    'O': 0,
    'B-KHABAR': 1,
    'I-KHABAR': 2,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_raw_text(text_path: str) -> str:
    """Charger le texte brut du corpus OpenITI."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def clean_openiti_text(text: str) -> str:
    """Nettoyer agressivement le texte OpenITI (supprimer tous les artefacts)."""
    # Supprimer les metadonnees completes
    text = re.sub(r'#META#.*?\n', '', text)
    # Supprimer les headers OpenITI
    text = re.sub(r'^#+OpenITI.*?\n', '', text, flags=re.MULTILINE)
    # Supprimer les separateurs de page
    text = re.sub(r'#Page[^\n]*\n', '', text)
    # Supprimer les headers de page
    text = re.sub(r'#PageV\d+P\d+\n', '', text)
    # Supprimer les markers de manuscrit
    text = re.sub(r'ms\d+', '', text)
    # Supprimer les continuations (~~ en debut de ligne)
    text = re.sub(r'^~~\s*', '', text, flags=re.MULTILINE)
    # Supprimer les signes diacritiques speciaux OpenITI
    text = re.sub(r'[©®™§]', '', text)
    # Supprimer les marqueurs de commentaires
    text = re.sub(r'\[.*?\]', '', text)
    # Normaliser les espaces multiples et newlines
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Diviser le texte en chunks chevauchants (sliding window) pour eviter fragmenter les akhbars."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        if i + chunk_size >= len(text):
            break
    return chunks


def extract_akhbars_from_text(
    text: str,
    model_path: str,
    tokenizer_name: str = 'aubmindlab/bert-base-arabertv2',
    confidence_threshold: float = 0.8
) -> List[Dict]:
    """
    Extraire les akhbars du texte brut en utilisant le modele fine-tune.
    Approche: traiter le texte en chunks chevauchants pour ne pas fragmenter les akhbars.
    """

    print(f"[*] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    # Pipeline pour token classification
    # IMPORTANT: pas d'aggregation_strategy pour voir les transitions entre akhbars
    nlp = pipeline(
        'token-classification',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    print(f"[*] Cleaning text agressively...")
    text = clean_openiti_text(text)
    print(f"    Text length after cleaning: {len(text)} characters")

    # Diviser en chunks chevauchants au lieu de phrases
    chunks = split_into_chunks(text, chunk_size=1500, overlap=300)
    print(f"    Split into {len(chunks)} overlapping chunks")

    akhbars = []
    seen_texts = set()  # Pour eviter les doublons dus au chevauchement

    print(f"[*] Extracting akhbars (confidence threshold: {confidence_threshold})...")

    for chunk_idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue

        # Predire les labels pour ce chunk
        try:
            predictions = nlp(chunk[:2000])  # Limiter la longueur
        except Exception as e:
            print(f"  [!] Error processing chunk {chunk_idx}: {e}")
            continue

        # Sans aggregation_strategy, on a les prédictions par TOKEN
        # On reconstruit les akhbars en groupant les tokens avec label B-KHABAR et I-KHABAR
        current_akhbar = []
        current_scores = []

        for pred in predictions:
            token = pred.get('word', '')
            entity = pred.get('entity', 'O')
            score = pred.get('score', 0.0)

            # Nettoyer les tokens AraBERT (## = continuation)
            if token.startswith('##'):
                if current_akhbar:
                    current_akhbar[-1] += token[2:]
                continue

            # B-KHABAR = début d'un nouvel akhbar
            if entity == 'B-KHABAR' and score >= confidence_threshold:
                # Sauvegarder l'akhbar précédent s'il existe
                if current_akhbar:
                    akhbar_text = ' '.join(current_akhbar)
                    if akhbar_text not in seen_texts and len(akhbar_text) > 20:
                        avg_score = np.mean(current_scores)
                        akhbars.append({
                            'text': akhbar_text,
                            'confidence': float(avg_score),
                            'length': len(current_akhbar)
                        })
                        seen_texts.add(akhbar_text)

                # Commencer un nouvel akhbar
                current_akhbar = [token]
                current_scores = [score]

            # I-KHABAR = continuation d'un akhbar
            elif entity == 'I-KHABAR' and score >= confidence_threshold and current_akhbar:
                current_akhbar.append(token)
                current_scores.append(score)

            # O = fin d'akhbar
            else:
                if current_akhbar:
                    akhbar_text = ' '.join(current_akhbar)
                    if akhbar_text not in seen_texts and len(akhbar_text) > 20:
                        avg_score = np.mean(current_scores)
                        akhbars.append({
                            'text': akhbar_text,
                            'confidence': float(avg_score),
                            'length': len(current_akhbar)
                        })
                        seen_texts.add(akhbar_text)
                    current_akhbar = []
                    current_scores = []

        # Sauvegarder le dernier akhbar du chunk
        if current_akhbar:
            akhbar_text = ' '.join(current_akhbar)
            if akhbar_text not in seen_texts and len(akhbar_text) > 20:
                avg_score = np.mean(current_scores)
                akhbars.append({
                    'text': akhbar_text,
                    'confidence': float(avg_score),
                    'length': len(current_akhbar)
                })
                seen_texts.add(akhbar_text)

    print(f"[OK] Extracted {len(akhbars)} unique akhbars")

    return akhbars


def compare_with_reference(
    extracted_akhbars: List[Dict],
    reference_json_path: str,
    output_path: str
) -> Dict:
    """
    Comparer les akhbars extraits avec les annotations de reference.
    """

    print(f"[*] Loading reference annotations from {reference_json_path}...")
    with open(reference_json_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    reference_akhbars = []
    for akh in ref_data['akhbar']:
        # Combiner tous les segments
        text = ' '.join([seg['text'] for seg in akh['content']['segments']])
        reference_akhbars.append({
            'text': text,
            'num': akh['num'],
            'subtype': akh.get('subtype', 'unknown')
        })

    print(f"[*] Comparing {len(extracted_akhbars)} extracted with {len(reference_akhbars)} reference akhbars...")

    # Statistiques simples
    results = {
        'extracted_count': len(extracted_akhbars),
        'reference_count': len(reference_akhbars),
        'extracted_tokens': sum(ak['length'] for ak in extracted_akhbars),
        'reference_tokens': sum(len(ak['text'].split()) for ak in reference_akhbars),
        'avg_confidence': float(np.mean([ak['confidence'] for ak in extracted_akhbars])) if extracted_akhbars else 0,
    }

    print(f"\n[RESULTS]")
    print(f"  Extracted: {results['extracted_count']} akhbars ({results['extracted_tokens']} tokens)")
    print(f"  Reference: {results['reference_count']} akhbars ({results['reference_tokens']} tokens)")
    print(f"  Avg confidence: {results['avg_confidence']:.3f}")

    # Sauvegarder les resultats
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'extracted_akhbars': extracted_akhbars[:100],  # Garder les 100 premiers pour inspection
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Results saved to {output_path}")

    return results


def main():
    # Chemins
    model_path = Path('./checkpoints/arabertv2_akhbars')
    raw_text_path = Path('./0406IbnHabibNaysaburi/0406IbnHabibNaysaburi.CuqalaMajanin/0406IbnHabibNaysaburi.CuqalaMajanin.Shamela0001593-ara1')
    reference_json_path = Path('./data/processed/Kitab_Uqala_al_Majanin_annotated.json')
    output_path = Path('./results/evaluation_results.json')

    # Verifier que le modele est pret
    if not model_path.exists():
        print(f"[!] Model not found at {model_path}")
        print("    Please wait for fine-tuning to complete.")
        return

    # Charger le texte brut
    print(f"[*] Loading raw text from {raw_text_path}...")
    raw_text = load_raw_text(str(raw_text_path))
    print(f"    Text length: {len(raw_text)} characters")

    # Extraire les akhbars
    extracted = extract_akhbars_from_text(
        raw_text,
        str(model_path),
        confidence_threshold=0.7
    )

    # Comparer avec les references
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = compare_with_reference(
        extracted,
        str(reference_json_path),
        str(output_path)
    )


if __name__ == '__main__':
    main()
