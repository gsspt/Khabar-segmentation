#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualiser les boundary tokens dans le corpus kitab_uqala_reference_corpus.txt
"""

import json
from pathlib import Path
from transformers import AutoTokenizer


def create_boundary_visualization(
    corpus_path: str,
    boundary_tokens_path: str,
    output_path: str,
    model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
) -> None:
    """
    Crée une visualisation HTML avec les boundary tokens surlignés.
    """

    print(f"[1/4] Charger le corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      Corpus: {len(text):,} caractères")

    print(f"\n[2/4] Charger les boundary tokens...")
    with open(boundary_tokens_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    boundary_indices = data['boundary_indices']
    boundary_count = len(boundary_indices)
    print(f"      Boundary tokens: {boundary_count:,}")

    print(f"\n[3/4] Tokeniser le corpus...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    print(f"      Tokens totaux: {len(tokens):,}")

    # Créer un set des indices boundary pour recherche rapide
    boundary_set = set(boundary_indices)

    print(f"\n[4/4] Générer la visualisation HTML...")

    # HTML header
    html = """<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualisation Boundary Tokens - Kitab Uqala</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap" rel="stylesheet">
    <style>
        body {
            direction: rtl;
            font-family: 'Noto Naskh Arabic', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            line-height: 2;
        }

        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
        }

        .info {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }

        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background-color: #fafafa;
            border-radius: 5px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 3px;
            background-color: rgba(255, 165, 0, 0.3);
            border: 2px solid #ff8c00;
        }

        .text-container {
            background-color: white;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            text-align: justify;
            word-wrap: break-word;
            font-size: 16px;
        }

        .boundary-token {
            background-color: rgba(255, 165, 0, 0.3);
            border-left: 3px solid #ff8c00;
            padding: 2px 4px;
            margin: 0 2px;
            border-radius: 2px;
        }

        .stats {
            margin-top: 20px;
            padding: 15px;
            background-color: #f0f0f0;
            border-radius: 5px;
            font-size: 14px;
        }

        .stats p {
            margin: 5px 0;
        }

        .stat-value {
            font-weight: bold;
            color: #ff8c00;
        }

        .footer {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Visualisation Boundary Tokens - Kitab Uqala</h1>

        <div class="info">
            Tokens surlignés en <span style="background-color: rgba(255, 165, 0, 0.3); padding: 2px 4px;">orange</span>
            = tokens classifiés comme boundary par le modèle CAMeL-BERT
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-color"></div>
                <span>Boundary Token</span>
            </div>
        </div>

        <div class="text-container">
"""

    # Construire le contenu du texte avec les tokens surlignés
    current_pos = 0

    for token_idx, token in enumerate(tokens):
        # Ajouter le token (avec surlignage si boundary)
        if token_idx in boundary_set:
            # C'est un boundary token
            html += f'<span class="boundary-token">{token}</span>'
        else:
            # Token normal
            html += token

        # Ajouter un espace après le token (sauf certains caractères spéciaux)
        if not token.startswith('##') and token not in ['۔', '،', '؟', '!']:
            html += ' '

    html += """
        </div>

        <div class="stats">
            <p><strong>Statistiques:</strong></p>
"""

    html += f'            <p>Corpus: <span class="stat-value">{len(text):,}</span> caractères</p>\n'
    html += f'            <p>Tokens totaux: <span class="stat-value">{len(tokens):,}</span></p>\n'
    html += f'            <p>Boundary tokens: <span class="stat-value">{boundary_count:,}</span></p>\n'
    html += f'            <p>Pourcentage: <span class="stat-value">{100 * boundary_count / len(tokens):.2f}%</span></p>\n'

    html += """
        </div>

        <div class="footer">
            <p>Généré par le modèle CAMeL-BERT (camelbert_binary_classification_final)</p>
            <p>Visualisation simple des boundary tokens - 2026-04-21</p>
        </div>
    </div>
</body>
</html>
"""

    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size = Path(output_path).stat().st_size / 1024
    print(f"\n✓ Visualisation générée: {output_path}")
    print(f"  Taille: {file_size:.1f} KB")
    print(f"\nRésumé:")
    print(f"  Corpus: {len(text):,} caractères")
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Boundary tokens: {boundary_count:,} ({100 * boundary_count / len(tokens):.2f}%)")


if __name__ == "__main__":
    corpus_path = "data/processed/kitab_uqala_reference_corpus.txt"
    boundary_tokens_path = "results/camelbert_boundary_tokens_clean.json"
    output_path = "results/visualization_boundary_tokens.html"

    create_boundary_visualization(
        corpus_path=corpus_path,
        boundary_tokens_path=boundary_tokens_path,
        output_path=output_path
    )
