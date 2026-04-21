#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualisation interactive avancée avec filtres et statistiques détaillées.
"""

import json
import re
from typing import List, Dict, Tuple, Set


def load_data(corpus_path: str, boundaries_path: str, camelbert_path: str):
    """Charger toutes les données nécessaires."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    with open(boundaries_path, 'r', encoding='utf-8') as f:
        gold_boundaries = json.load(f)

    with open(camelbert_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    camelbert_segments = data.get('segmentation', {}).get('segments', [])

    return text, gold_boundaries, camelbert_segments


def find_overlaps(range1: Tuple[int, int], ranges_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Trouver les chevauchements entre une plage et une liste de plages."""
    overlaps = []
    for range2 in ranges_list:
        overlap_start = max(range1[0], range2[0])
        overlap_end = min(range1[1], range2[1])
        if overlap_start < overlap_end:
            overlaps.append((overlap_start, overlap_end))
    return overlaps


def calculate_metrics(gold_boundaries: List[Dict], camelbert_segments: List[Dict]) -> Dict:
    """Calculer les métriques de comparaison."""
    gold_ranges = [(b['start'], b['end']) for b in gold_boundaries]

    camelbert_isnad = [(s['start'], s['end']) for s in camelbert_segments if s.get('type') == 'isnad']
    camelbert_prose = [(s['start'], s['end']) for s in camelbert_segments if s.get('type') == 'prose']

    # Chevauchement des isnads CAMeL-BERT avec le gold standard
    total_overlap = 0
    for isnad_range in camelbert_isnad:
        overlaps = find_overlaps(isnad_range, gold_ranges)
        total_overlap += sum(end - start for start, end in overlaps)

    total_isnad_chars = sum(end - start for start, end in camelbert_isnad)

    metrics = {
        'gold_standard': {
            'count': len(gold_ranges),
            'total_chars': sum(end - start for start, end in gold_ranges),
        },
        'camelbert': {
            'segments': len(camelbert_segments),
            'isnads': len(camelbert_isnad),
            'prose': len(camelbert_prose),
            'isnad_chars': total_isnad_chars,
            'overlap_with_gold': total_overlap,
        }
    }

    if total_isnad_chars > 0:
        metrics['camelbert']['overlap_percentage'] = (total_overlap / total_isnad_chars) * 100

    return metrics


def create_interactive_html(text: str, gold_boundaries: List[Dict], camelbert_segments: List[Dict],
                            metrics: Dict, output_path: str) -> None:
    """Créer une visualisation interactive avancée."""

    html = """<!DOCTYPE html>
<html dir='rtl' lang='ar'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Visualisation Interactive - Kitab Uqala</title>
    <link href='https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap' rel='stylesheet'>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            direction: rtl;
            font-family: 'Noto Naskh Arabic', Arial, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .header h1 {
            margin: 0;
            color: #333;
            text-align: center;
            font-size: 28px;
        }

        .controls {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }

        .control-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .control-group label {
            font-weight: bold;
            color: #333;
        }

        .checkbox {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .metrics-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 16px;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }

        .metric-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }

        .text-container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            line-height: 2.2;
            word-wrap: break-word;
            text-align: justify;
            font-size: 16px;
        }

        .legend {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
            border: 2px solid;
        }

        .gold-boundary { background-color: rgba(0, 0, 255, 0.25); border-left: 3px solid #0000ff; }
        .camelbert-isnad { background-color: rgba(255, 165, 0, 0.25); border-left: 3px solid #ffa500; }
        .camelbert-prose { opacity: 0.5; }

        .comparison-summary {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .comparison-summary h2 {
            margin-top: 0;
            color: #667eea;
        }

        .summary-text {
            line-height: 1.6;
            color: #666;
        }

        .footer {
            text-align: center;
            color: white;
            margin-top: 30px;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class='main-container'>
        <div class='header'>
            <h1>📚 Visualisation Interactive de la Segmentation - Kitab Uqala</h1>
        </div>

        <div class='legend'>
            <div class='legend-item'>
                <div class='legend-color' style='background-color: rgba(0, 0, 255, 0.25); border-color: #0000ff;'></div>
                <span>Gold Standard (Boundaries Vraies)</span>
            </div>
            <div class='legend-item'>
                <div class='legend-color' style='background-color: rgba(255, 165, 0, 0.25); border-color: #ffa500;'></div>
                <span>CAMeL-BERT Isnads Détectés</span>
            </div>
        </div>

        <div class='metrics-panel'>
            <div class='metric-card'>
                <h3>Gold Standard</h3>
                <div class='metric-value'>""" + str(metrics['gold_standard']['count']) + """</div>
                <div class='metric-label'>Boundaries (khabars)</div>
                <div class='metric-value' style='font-size: 16px; margin-top: 10px;'>""" + str(metrics['gold_standard']['total_chars']) + """</div>
                <div class='metric-label'>Caractères</div>
            </div>

            <div class='metric-card'>
                <h3>CAMeL-BERT Segments</h3>
                <div class='metric-value'>""" + str(metrics['camelbert']['segments']) + """</div>
                <div class='metric-label'>Segments totaux</div>
                <div style='margin-top: 10px; font-size: 13px; color: #666;'>
                    <strong>Isnads:</strong> """ + str(metrics['camelbert']['isnads']) + """ |
                    <strong>Prose:</strong> """ + str(metrics['camelbert']['prose']) + """
                </div>
            </div>

            <div class='metric-card'>
                <h3>Chevauchement</h3>
                <div class='metric-value' style='font-size: 18px; color: #ff6b6b;'>""" + str(round(metrics['camelbert'].get('overlap_percentage', 0), 1)) + """%</div>
                <div class='metric-label'>Isnads CAMeL-BERT qui chevauchent le gold standard</div>
            </div>
        </div>

        <div class='controls'>
            <div class='control-group'>
                <label>Affichage:</label>
                <label><input type='checkbox' class='checkbox' id='show-gold' checked> Gold Standard</label>
                <label><input type='checkbox' class='checkbox' id='show-camelbert' checked> CAMeL-BERT Isnads</label>
            </div>
        </div>

        <div class='text-container' id='text-content'>
            <!-- Texte avec surlignage généré -->
        </div>

        <div class='comparison-summary'>
            <h2>Analyse Comparative</h2>
            <div class='summary-text'>
                <p><strong>Gold Standard:</strong> """ + str(metrics['gold_standard']['count']) + """ boundaries (khabars) annotés manuellement</p>
                <p><strong>CAMeL-BERT:</strong> """ + str(metrics['camelbert']['segments']) + """ segments détectés (""" + str(metrics['camelbert']['isnads']) + """ isnads + """ + str(metrics['camelbert']['prose']) + """ prose)</p>
                <p><strong>Chevauchement:</strong> """ + str(round(metrics['camelbert'].get('overlap_percentage', 0), 1)) + """% des caractères des isnads CAMeL-BERT chevauchent les boundaries du gold standard</p>
            </div>
        </div>

        <div class='footer'>
            <p>Visualisation générée le 2026-04-21 | Corpus: Kitab Uqala (""" + str(len(text)) + """ caractères)</p>
        </div>
    </div>

    <script>
        const textContent = `""" + text.replace('`', '\\`').replace('\\', '\\\\') + """`;
        const goldBoundaries = """ + json.dumps([(b['start'], b['end']) for b in gold_boundaries]) + """;
        const camelBertSegments = """ + json.dumps([(s['start'], s['end'], s.get('type', '')) for s in camelbert_segments]) + """;

        function renderText() {
            const showGold = document.getElementById('show-gold').checked;
            const showCamelBert = document.getElementById('show-camelbert').checked;

            let html = '';
            let lastPos = 0;
            let markers = [];

            // Construire les markers
            if (showGold) {
                goldBoundaries.forEach(([start, end]) => {
                    markers.push({pos: start, type: 'gold-start'});
                    markers.push({pos: end, type: 'gold-end'});
                });
            }

            if (showCamelBert) {
                camelBertSegments.forEach(([start, end, type]) => {
                    if (type === 'isnad') {
                        markers.push({pos: start, type: 'isnad-start'});
                        markers.push({pos: end, type: 'isnad-end'});
                    }
                });
            }

            // Trier les markers
            markers.sort((a, b) => a.pos - b.pos);

            let activeClasses = new Set();
            let markerIdx = 0;

            for (let i = 0; i < textContent.length && markerIdx < markers.length; i++) {
                if (markers[markerIdx].pos === i) {
                    // Ajouter le texte jusqu'ici
                    if (i > lastPos) {
                        const chunk = textContent.substring(lastPos, i);
                        if (activeClasses.size > 0) {
                            html += '<span class="' + Array.from(activeClasses).join(' ') + '">' + escapeHtml(chunk) + '</span>';
                        } else {
                            html += escapeHtml(chunk);
                        }
                    }

                    // Traiter tous les markers à cette position
                    while (markerIdx < markers.length && markers[markerIdx].pos === i) {
                        const marker = markers[markerIdx];
                        if (marker.type.includes('start')) {
                            if (marker.type === 'gold-start') activeClasses.add('gold-boundary');
                            if (marker.type === 'isnad-start') activeClasses.add('camelbert-isnad');
                        } else {
                            if (marker.type === 'gold-end') activeClasses.delete('gold-boundary');
                            if (marker.type === 'isnad-end') activeClasses.delete('camelbert-isnad');
                        }
                        markerIdx++;
                    }

                    lastPos = i;
                }
            }

            // Ajouter le reste du texte
            if (lastPos < textContent.length) {
                const remaining = textContent.substring(lastPos);
                if (activeClasses.size > 0) {
                    html += '<span class="' + Array.from(activeClasses).join(' ') + '">' + escapeHtml(remaining) + '</span>';
                } else {
                    html += escapeHtml(remaining);
                }
            }

            document.getElementById('text-content').innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Ajouter les event listeners
        document.getElementById('show-gold').addEventListener('change', renderText);
        document.getElementById('show-camelbert').addEventListener('change', renderText);

        // Rendu initial
        renderText();
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Visualisation interactive générée: {output_path}")


def main():
    corpus_path = "data/processed/kitab_uqala_reference_corpus.txt"
    boundaries_path = "data/processed/kitab_uqala_boundaries.json"
    camelbert_path = "results/camelbert_kitab_uqala_segments.json"
    output_path = "results/visualization_segmentation_interactive.html"

    print("Chargement des données...")
    text, gold_boundaries, camelbert_segments = load_data(corpus_path, boundaries_path, camelbert_path)

    print("Calcul des métriques...")
    metrics = calculate_metrics(gold_boundaries, camelbert_segments)

    print("Création de la visualisation interactive...")
    create_interactive_html(text, gold_boundaries, camelbert_segments, metrics, output_path)

    print(f"\nStatistiques:")
    print(f"  Gold Standard: {metrics['gold_standard']['count']} boundaries")
    print(f"  CAMeL-BERT: {metrics['camelbert']['segments']} segments")
    print(f"  Chevauchement des isnads: {metrics['camelbert'].get('overlap_percentage', 0):.1f}%")


if __name__ == "__main__":
    main()
