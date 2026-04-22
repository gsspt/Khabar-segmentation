#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation HTML du texte brut avec isnads détectés par CAMeL-BERT surlignés
"""

import json
from pathlib import Path


def visualize_detected_isnads():
    """Crée une visualisation HTML des isnads détectés par CAMeL-BERT."""

    print("[1/4] Charger les données...")

    # Charger le texte brut
    with open('data/processed/kitab_uqala_reference_corpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"      Texte: {len(text):,} caractères")

    # Charger les clusters CAMeL-BERT
    with open('results/camelbert_char_boundaries_v2.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    boundaries = data.get('khabar_boundaries', [])
    metadata = data.get('metadata', {})

    print(f"      Clusters détectés: {len(boundaries)}")
    print(f"      F1 Score: {metadata.get('evaluation_tol80', {}).get('f1', 'N/A')}")

    print("\n[2/4] Construire la visualisation HTML...")

    # Créer les positions à surligner
    positions_to_highlight = []
    for boundary in boundaries:
        start = boundary['char_start']
        end = boundary['char_end']
        positions_to_highlight.append({
            'start': start,
            'end': end,
            'boundary_id': boundary['boundary_id'],
            'n_tokens': boundary['n_tokens'],
        })

    # Trier par position
    positions_to_highlight.sort(key=lambda x: x['start'])

    # Construire le HTML
    html_parts = []
    last_pos = 0

    for pos_info in positions_to_highlight:
        start = pos_info['start']
        end = pos_info['end']
        boundary_id = pos_info['boundary_id']
        n_tokens = pos_info['n_tokens']

        # Ajouter le texte normal avant le cluster
        if last_pos < start:
            normal_text = text[last_pos:start]
            # Échapper les caractères spéciaux HTML
            normal_text = (normal_text
                          .replace('&', '&amp;')
                          .replace('<', '&lt;')
                          .replace('>', '&gt;'))
            html_parts.append(normal_text)

        # Ajouter le cluster surligné
        cluster_text = text[start:end]
        cluster_text_escaped = (cluster_text
                               .replace('&', '&amp;')
                               .replace('<', '&lt;')
                               .replace('>', '&gt;'))

        # Créer un span avec un ID et des attributs data
        html_parts.append(
            f'<span class="isnad-cluster" data-id="{boundary_id}" '
            f'data-start="{start}" data-end="{end}" data-tokens="{n_tokens}" '
            f'title="Cluster #{boundary_id}: {n_tokens} tokens">'
            f'{cluster_text_escaped}</span>'
        )

        last_pos = end

    # Ajouter le reste du texte
    if last_pos < len(text):
        remaining_text = text[last_pos:]
        remaining_text = (remaining_text
                         .replace('&', '&amp;')
                         .replace('<', '&lt;')
                         .replace('>', '&gt;'))
        html_parts.append(remaining_text)

    html_text = ''.join(html_parts)

    # Créer le document HTML complet
    html = f"""<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAMeL-BERT Isnads Detection - Kitab Uqala</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            direction: rtl;
            font-family: 'Noto Naskh Arabic', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1200px;
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
            color: #333;
            text-align: center;
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header p {{
            text-align: center;
            color: #666;
            font-size: 16px;
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
            justify-content: space-between;
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
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            background: rgba(255, 107, 107, 0.3);
            border: 2px solid #ff6b6b;
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
            padding: 15px;
            background: #f9f9f9;
            border-left: 4px solid #ff6b6b;
            border-radius: 3px;
        }}

        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #ff6b6b;
        }}

        .stat-unit {{
            font-size: 14px;
            color: #999;
            margin-top: 4px;
        }}

        .text-container {{
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            line-height: 2.2;
            word-wrap: break-word;
            text-align: justify;
            font-size: 18px;
            color: #333;
            margin-bottom: 20px;
        }}

        .isnad-cluster {{
            background-color: rgba(255, 107, 107, 0.25);
            border-left: 3px solid #ff6b6b;
            padding: 2px 4px;
            margin: 0 1px;
            border-radius: 2px;
            cursor: pointer;
            transition: all 0.2s ease;
            position: relative;
        }}

        .isnad-cluster:hover {{
            background-color: rgba(255, 107, 107, 0.45);
            border-left-color: #ff4444;
            box-shadow: 0 0 8px rgba(255, 107, 107, 0.5);
            border-radius: 3px;
        }}

        .info-box {{
            position: fixed;
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            z-index: 1000;
            display: none;
            max-width: 400px;
            font-size: 14px;
        }}

        .info-box.visible {{
            display: block;
        }}

        .info-box-title {{
            font-weight: bold;
            color: #ff6b6b;
            margin-bottom: 8px;
        }}

        .info-box-content {{
            color: #666;
            line-height: 1.6;
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

        .toggle-controls {{
            display: flex;
            gap: 10px;
        }}

        button {{
            padding: 8px 16px;
            background: #ff6b6b;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}

        button:hover {{
            background: #ff4444;
        }}

        button.inactive {{
            background: #ccc;
            cursor: not-allowed;
        }}

        @media (max-width: 768px) {{
            .text-container {{
                padding: 20px;
                font-size: 16px;
            }}

            .header h1 {{
                font-size: 24px;
            }}

            .controls {{
                flex-direction: column;
                align-items: flex-start;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 CAMeL-BERT Isnad Detection</h1>
            <p>Kitab al-'Uqala' al-Majanin - Isnads surlignés</p>
        </div>

        <div class="controls">
            <div class="legend">
                <span style="font-weight: bold; color: #333;">Légende:</span>
                <div class="legend-item">
                    <div class="legend-color"></div>
                    <span>Isnad détecté par CAMeL-BERT</span>
                </div>
            </div>
            <div class="toggle-controls">
                <button id="highlight-toggle">Activer surlignes</button>
                <button id="stats-toggle">Afficher stats</button>
            </div>
        </div>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Corpus</div>
                <div class="stat-value">{len(text):,}</div>
                <div class="stat-unit">caractères</div>
            </div>
            <div class="stat">
                <div class="stat-label">Isnads Détectés</div>
                <div class="stat-value">{len(boundaries)}</div>
                <div class="stat-unit">clusters</div>
            </div>
            <div class="stat">
                <div class="stat-label">F1 Score</div>
                <div class="stat-value">{metadata.get('evaluation_tol80', {}).get('f1', 'N/A')}</div>
                <div class="stat-unit">vs gold standard</div>
            </div>
            <div class="stat">
                <div class="stat-label">Precision</div>
                <div class="stat-value">{metadata.get('evaluation_tol80', {}).get('precision', 'N/A'):.2%}</div>
                <div class="stat-unit">peu de faux positifs</div>
            </div>
        </div>

        <div class="text-container" id="text-container">
            {html_text}
        </div>

        <div class="footer">
            <p><strong>Source:</strong> results/camelbert_char_boundaries_v2.json</p>
            <p><strong>Model:</strong> {metadata.get('model', 'CAMeL-BERT')}</p>
            <p><strong>Method:</strong> {metadata.get('method', 'Direct char extraction')}</p>
            <p>Visualisation générée: 2026-04-22</p>
        </div>
    </div>

    <div class="info-box" id="info-box">
        <div class="info-box-title">Isnad Détecté</div>
        <div class="info-box-content" id="info-box-content"></div>
    </div>

    <script>
        // Gestion de la visualisation
        const clusters = {json.dumps(positions_to_highlight, ensure_ascii=False)};
        const corpus = `{text[:50000]}...`;  // Stockage partiel pour démo
        let highlightingEnabled = true;

        // Toggle surlignes
        document.getElementById('highlight-toggle').addEventListener('click', function() {{
            highlightingEnabled = !highlightingEnabled;
            const clusters = document.querySelectorAll('.isnad-cluster');
            if (highlightingEnabled) {{
                clusters.forEach(c => c.style.opacity = '1');
                this.textContent = 'Désactiver surlignes';
                this.classList.remove('inactive');
            }} else {{
                clusters.forEach(c => c.style.opacity = '0.2');
                this.textContent = 'Activer surlignes';
                this.classList.add('inactive');
            }}
        }});

        // Info box au survol
        document.querySelectorAll('.isnad-cluster').forEach(cluster => {{
            cluster.addEventListener('mouseenter', function(e) {{
                const id = this.getAttribute('data-id');
                const start = this.getAttribute('data-start');
                const end = this.getAttribute('data-end');
                const tokens = this.getAttribute('data-tokens');
                const text = this.textContent;

                const infoBox = document.getElementById('info-box');
                const content = document.getElementById('info-box-content');

                content.innerHTML = `
                    <p><strong>ID:</strong> #${{id}}</p>
                    <p><strong>Position:</strong> [${{start}}:${{end}}]</p>
                    <p><strong>Tokens:</strong> ${{tokens}}</p>
                    <p><strong>Texte:</strong> <em>"${{text.substring(0, 100)}}..."</em></p>
                `;

                infoBox.classList.add('visible');
                infoBox.style.top = (e.clientY + 10) + 'px';
                infoBox.style.left = (e.clientX + 10) + 'px';
            }});

            cluster.addEventListener('mouseleave', function() {{
                document.getElementById('info-box').classList.remove('visible');
            }});
        }});

        // Suivi du mouvement de la souris pour la boîte d'info
        document.addEventListener('mousemove', function(e) {{
            const infoBox = document.getElementById('info-box');
            if (infoBox.classList.contains('visible')) {{
                infoBox.style.top = (e.clientY + 10) + 'px';
                infoBox.style.left = (e.clientX + 10) + 'px';
            }}
        }});

        // Scroll au cluster
        function scrollToCluster(id) {{
            const cluster = document.querySelector(`[data-id="${{id}}"]`);
            if (cluster) {{
                cluster.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                cluster.style.animation = 'highlight 2s';
            }}
        }}

        // Statistiques de défilement
        let scrollCount = 0;
        window.addEventListener('scroll', function() {{
            scrollCount++;
            if (scrollCount % 50 === 0) {{
                // Optionnel: envoyer des métriques
            }}
        }});
    </script>
</body>
</html>"""

    # Sauvegarder
    output_path = 'results/visualization_detected_isnads.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"[OK] Visualisation generee: {output_path}")
    print(f"  Taille: {file_size:.1f} MB")

    print(f"\n[4/4] Resume")
    print(f"  Corpus: {len(text):,} caracteres")
    print(f"  Isnads surlignés: {len(boundaries)}")
    print(f"  F1 Score: {metadata.get('evaluation_tol80', {}).get('f1', 'N/A')}")


if __name__ == "__main__":
    visualize_detected_isnads()
