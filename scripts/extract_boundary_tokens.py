#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrait les tokens boundary (isnad tokens) détectés par CAMeL-BERT
et crée un fichier propre avec la liste des tokens prédits comme isnads.

A EXÉCUTER SUR GOOGLE COLAB (où PyTorch/Transformers est installé)
"""

import json
from transformers import AutoTokenizer
import sys


def extract_boundary_tokens(
    corpus_path: str,
    predictions_path: str,
    output_path: str,
    model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
) -> None:
    """
    Extrait les tokens boundary du corpus basé sur les prédictions du modèle.

    Args:
        corpus_path: Chemin vers le fichier texte du corpus
        predictions_path: Chemin vers le fichier JSON avec les prédictions
        output_path: Chemin où sauvegarder les résultats
        model_name: Nom du modèle CAMeL-BERT sur Hugging Face
    """

    print(f"[1/4] Charger le tokenizer {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("      ✓ Tokenizer chargé")
    except Exception as e:
        print(f"      ✗ Erreur: {e}")
        print("      Assurez-vous d'être sur Google Colab avec PyTorch installé")
        sys.exit(1)

    print(f"\n[2/4] Charger le corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      ✓ Corpus chargé: {len(text):,} caractères")

    print(f"\n[3/4] Tokenizer le corpus...")
    tokens = tokenizer.tokenize(text)
    print(f"      ✓ {len(tokens):,} tokens générés")

    print(f"\n[4/4] Charger les prédictions du modèle...")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)

    predictions = predictions_data['inference_results']['predictions']
    print(f"      ✓ {len(predictions):,} prédictions chargées")

    # Vérifier la correspondance
    if len(tokens) != len(predictions):
        print(f"\n⚠️  Attention: Mismatch détecté!")
        print(f"   Tokens: {len(tokens)}")
        print(f"   Prédictions: {len(predictions)}")
        print(f"   Différence: {abs(len(predictions) - len(tokens))}")

        # Utiliser la taille minimale
        min_len = min(len(tokens), len(predictions))
        tokens = tokens[:min_len]
        predictions = predictions[:min_len]
        print(f"   Utilisation de {min_len} pour l'extraction")

    # Extraire les boundary tokens
    print(f"\nExtraction des boundary tokens...")
    boundary_tokens = []
    boundary_indices = []

    for idx, (token, prediction) in enumerate(zip(tokens, predictions)):
        if prediction == 1:  # isnad token
            boundary_tokens.append(token)
            boundary_indices.append(idx)

    print(f"✓ {len(boundary_tokens):,} boundary tokens trouvés ({len(boundary_indices):,} indices)")

    # Créer les résultats
    results = {
        'metadata': {
            'corpus': corpus_path,
            'corpus_size_chars': len(text),
            'corpus_size_tokens': len(tokens),
            'model': model_name,
            'total_boundary_tokens': len(boundary_tokens),
            'boundary_percentage': round((len(boundary_tokens) / len(tokens)) * 100, 2),
        },
        'boundary_tokens': boundary_tokens,
        'boundary_indices': boundary_indices,
    }

    # Sauvegarder
    print(f"\nSauvegarde des résultats dans {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ Fichier sauvegardé")

    # Afficher un aperçu
    print(f"\n{'='*80}")
    print(f"RÉSUMÉ")
    print(f"{'='*80}")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Boundary tokens (isnad): {len(boundary_tokens):,}")
    print(f"Pourcentage: {results['metadata']['boundary_percentage']}%")
    print(f"\nPremiers 30 boundary tokens:")
    for i, token in enumerate(boundary_tokens[:30], 1):
        print(f"  {i:3d}. {token}")

    print(f"\n... (affiche les 30 premiers sur {len(boundary_tokens):,})")
    print(f"\nFichier complet créé: {output_path}")


if __name__ == "__main__":
    # Configuration
    CORPUS_PATH = "data/processed/kitab_uqala_reference_corpus.txt"
    PREDICTIONS_PATH = "results/camelbert_kitab_uqala_raw_inference.json"
    OUTPUT_PATH = "results/camelbert_boundary_tokens_clean.json"

    extract_boundary_tokens(
        corpus_path=CORPUS_PATH,
        predictions_path=PREDICTIONS_PATH,
        output_path=OUTPUT_PATH
    )
