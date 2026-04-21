#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation locale simple (aucune dépendance PyTorch)
Utilise le JSON avec les offsets générés par Colab
"""

import json
from pathlib import Path


def visualize(corpus_path: str, offsets_json_path: str, output_html: str):
    """Crée la visualisation HTML à partir des offsets."""

    print("[1/3] Charger le corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      {len(text):,} caractères")

    print("\n[2/3] Charger les boundary offsets...")
    with open(offsets_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    offsets = data['boundary_offsets']
    metadata = data['metadata']
    print(f"      {len(offsets):,} boundary tokens")

    # Créer l'ensemble des positions à surligner
    print("\n[3/3] Créer la visualisation HTML...")
    char_to_highlight = set()
    for offset in offsets:
        for char_pos in range(offset['char_start'], offset['char_end']):
            char_to_highlight.add(char_pos)

    # Construire le HTML
    html_text = ""
    for char_idx, char in enumerate(text):
        if char_idx in char_to_highlight:
            # Surligner ce caractère
            html_text += f'<span class="boundary">{char}</span>'
        else:
            # Échapper les caractères HTML
            if char == '&':
                html_text += '&amp;'
            elif char == '<':
                html_text += '&lt;'
            elif char == '>':
                html_text += '&gt;'
            else:
                html_text += char

    # Template HTML
    html = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Boundary Tokens - Kitab Uqala</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            direction: rtl;
            font-family: 'Noto Naskh Arabic', Arial, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin: 0;
            color: #333;
            text-align: center;
            font-size: 28px;
        }}

        .info {{
            background: white;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            color: #666;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .stat {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #ff8c00;
        }}

        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ff8c00;
        }}

        .text {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            line-height: 2;
            word-wrap: break-word;
            text-align: justify;
            font-size: 16px;
            color: #333;
            margin-bottom: 20px;
        }}

        .boundary {{
            background-color: rgba(255, 165, 0, 0.3);
            border-left: 2px solid #ff8c00;
            padding: 1px 2px;
            cursor: pointer;
        }}

        .boundary:hover {{
            background-color: rgba(255, 165, 0, 0.5);
        }}

        .footer {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #999;
            font-size: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .footer p {{
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Boundary Tokens - Kitab Uqala</h1>
        </div>

        <div class="info">
            Caractères en <span style="background: rgba(255, 165, 0, 0.3); padding: 2px 4px;">orange</span> = boundary tokens détectés par CAMeL-BERT
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Corpus</div>
                <div class="stat-value">{len(text):,}</div>
                <div class="stat-label">caractères</div>
            </div>
            <div class="stat">
                <div class="stat-label">Boundary Tokens</div>
                <div class="stat-value">{len(offsets):,}</div>
                <div class="stat-label">détectés</div>
            </div>
            <div class="stat">
                <div class="stat-label">Caractères Surlignés</div>
                <div class="stat-value">{len(char_to_highlight):,}</div>
                <div class="stat-label">positions</div>
            </div>
            <div class="stat">
                <div class="stat-label">Pourcentage</div>
                <div class="stat-value">{metadata['boundary_percentage']:.2f}%</div>
                <div class="stat-label">du corpus</div>
            </div>
        </div>

        <div class="text">
            {html_text}
        </div>

        <div class="footer">
            <p><strong>Source:</strong> {Path(offsets_json_path).name}</p>
            <p><strong>Model:</strong> {metadata['model']}</p>
            <p>Generated: 2026-04-21</p>
        </div>
    </div>
</body>
</html>"""

    # Sauvegarder
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size = Path(output_html).stat().st_size / 1024
    print(f"\n✓ Visualisation générée: {output_html}")
    print(f"  Taille: {file_size:.1f} KB")
    print(f"\nStats:")
    print(f"  Corpus: {len(text):,} caractères")
    print(f"  Boundary tokens: {len(offsets):,}")
    print(f"  Caractères surlignés: {len(char_to_highlight):,}")


if __name__ == "__main__":
    corpus_path = "data/processed/kitab_uqala_reference_corpus.txt"
    offsets_json_path = "results/camelbert_boundary_tokens_with_offsets.json"
    output_html = "results/visualization_boundary_tokens_final.html"

    visualize(corpus_path, offsets_json_path, output_html)
