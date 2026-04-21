#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation simple des boundary tokens dans le corpus.
Utilise directement les tokens du fichier JSON (pas de retokenization).
"""

import json
import re
from pathlib import Path


def visualize_boundary_tokens(
    corpus_path: str,
    boundary_json_path: str,
    output_path: str
) -> None:
    """
    Crée une visualisation HTML en surlignant les boundary tokens.
    """

    print("[1/3] Charger le corpus...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      Corpus: {len(text):,} caractères")

    print("\n[2/3] Charger les boundary tokens...")
    with open(boundary_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    boundary_tokens = data['boundary_tokens']
    metadata = data['metadata']

    print(f"      Boundary tokens: {len(boundary_tokens):,}")
    print(f"      Corpus size (JSON): {metadata['corpus_size_chars']:,} chars")
    print(f"      Tokens total (JSON): {metadata['corpus_size_tokens']:,}")

    # Créer des patterns regex pour chercher les tokens
    # Les tokens contiennent parfois des caractères spéciaux, donc il faut les échapper
    print("\n[3/3] Générer la visualisation...")

    # Créer une version du texte où on va ajouter des balises HTML
    # On va itérer sur le texte et chercher les boundary tokens

    # Trier les tokens par longueur décroissante pour éviter les chevauchements partiels
    sorted_tokens = sorted(set(boundary_tokens), key=len, reverse=True)

    # Créer les positions à surligner
    positions_to_highlight = []

    for token in sorted_tokens:
        # Chercher toutes les occurrences du token
        pattern = re.escape(token)
        for match in re.finditer(pattern, text):
            start, end = match.span()
            positions_to_highlight.append((start, end, token))

    # Trier par position
    positions_to_highlight.sort(key=lambda x: x[0])

    # Fusionner les positions qui se chevauchent
    merged_positions = []
    for start, end, token in positions_to_highlight:
        if merged_positions and start < merged_positions[-1][1]:
            # Chevauchement - fusionner
            merged_positions[-1] = (merged_positions[-1][0], max(merged_positions[-1][1], end))
        else:
            merged_positions.append((start, end))

    print(f"      Positions trouvées: {len(merged_positions)}")

    # Construire le texte avec les balises HTML
    html_text = ""
    last_pos = 0

    for start, end in merged_positions:
        # Ajouter le texte normal
        html_text += text[last_pos:start].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Ajouter le token surligné
        html_text += f'<span class="boundary-token">{text[start:end].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")}</span>'
        last_pos = end

    # Ajouter le reste du texte
    html_text += text[last_pos:].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    # HTML template
    html = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Boundary Tokens Visualization - Kitab Uqala</title>
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

        .controls {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .legend {{
            display: flex;
            gap: 15px;
            align-items: center;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #333;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            background-color: rgba(255, 165, 0, 0.3);
            border: 2px solid #ff8c00;
        }}

        .stats {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}

        .stat {{
            padding: 10px;
            background: #f9f9f9;
            border-left: 4px solid #ff8c00;
            border-radius: 3px;
        }}

        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #ff8c00;
        }}

        .text-container {{
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

        .boundary-token {{
            background-color: rgba(255, 165, 0, 0.3);
            border-left: 3px solid #ff8c00;
            padding: 2px 4px;
            margin: 0 1px;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .boundary-token:hover {{
            background-color: rgba(255, 165, 0, 0.5);
            border-left-color: #ff6600;
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
            <h1>📚 Boundary Tokens Visualization - Kitab Uqala</h1>
        </div>

        <div class="controls">
            <div class="legend">
                <span style="font-weight: bold;">Légende:</span>
                <div class="legend-item">
                    <div class="legend-color"></div>
                    <span>Boundary Token (CAMeL-BERT)</span>
                </div>
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Corpus Size</div>
                <div class="stat-value">{len(text):,}</div>
                <div class="stat-label">caractères</div>
            </div>
            <div class="stat">
                <div class="stat-label">Boundary Tokens</div>
                <div class="stat-value">{len(boundary_tokens):,}</div>
                <div class="stat-label">tokens détectés</div>
            </div>
            <div class="stat">
                <div class="stat-label">Highlighted</div>
                <div class="stat-value">{len(merged_positions):,}</div>
                <div class="stat-label">positions</div>
            </div>
            <div class="stat">
                <div class="stat-label">Percentage</div>
                <div class="stat-value">{metadata['boundary_percentage']:.2f}%</div>
                <div class="stat-label">du corpus</div>
            </div>
        </div>

        <div class="text-container">
            {html_text}
        </div>

        <div class="footer">
            <p><strong>Source:</strong> {Path(boundary_json_path).name}</p>
            <p><strong>Model:</strong> {metadata['model']}</p>
            <p>Generated: 2026-04-21</p>
        </div>
    </div>
</body>
</html>"""

    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size = Path(output_path).stat().st_size / 1024
    print(f"\n✓ Visualisation générée: {output_path}")
    print(f"  Taille: {file_size:.1f} KB")
    print(f"\nRésumé:")
    print(f"  Corpus: {len(text):,} caractères")
    print(f"  Boundary tokens (JSON): {len(boundary_tokens):,}")
    print(f"  Positions surlignées: {len(merged_positions):,}")
    print(f"  Pourcentage: {metadata['boundary_percentage']}%")


if __name__ == "__main__":
    corpus_path = "data/processed/kitab_uqala_reference_corpus.txt"
    boundary_json_path = "results/camelbert_boundary_tokens_clean.json"
    output_path = "results/visualization_boundary_tokens_local.html"

    visualize_boundary_tokens(
        corpus_path=corpus_path,
        boundary_json_path=boundary_json_path,
        output_path=output_path
    )
