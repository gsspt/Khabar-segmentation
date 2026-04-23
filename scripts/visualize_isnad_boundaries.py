#!/usr/bin/env python3
"""
Visualize isnad boundaries from three segmentation pipelines.

Shows the raw text with color-coded boundary markers for:
- Deepseek: LLM semantic boundaries (green)
- CAMeL-BERT: Token classification boundaries (red)
- Baseline: Linguistic rule boundaries (blue)

Each boundary is marked inline with a visual indicator showing where units begin.

Usage:
    python scripts/visualize_isnad_boundaries.py \
      --text data/processed/ibnjawzi_hamqa_clean.txt \
      --deepseek results/IbnJawzi/gold_standard_ibnjawzi_deepseek.json \
      --camelbert results/IbnJawzi/camelbert_ibnjawzi_narrative_units.json \
      --baseline results/IbnJawzi/baseline_v4_ibnjawzi_narrative_units.json \
      --output results/visualization_ibnjawzi_boundaries.html
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_text(text_file: str) -> str:
    """Load text from file."""
    with open(text_file, 'r', encoding='utf-8') as f:
        return f.read()


def load_units(units_file: str) -> list[dict]:
    """Load narrative units from JSON."""
    with open(units_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('narrative_units', data.get('units', []))


def extract_boundaries(units: list[dict], pipeline_name: str) -> list[dict]:
    """
    Extract boundary positions from narrative units.

    Each unit's char_start represents where a new narrative/isnad begins.
    Returns list of {position, pipeline, unit_id, isnad_text}
    """
    boundaries = []
    for idx, unit in enumerate(units):
        start = unit.get('char_start', 0)
        isnad_text = unit.get('isnad_text')
        text = unit.get('text')

        # Handle None values
        if isnad_text is None:
            isnad_text = ''
        if text is None:
            text = ''

        isnad = (isnad_text or text)[:100]  # First 100 chars
        boundaries.append({
            'position': start,
            'pipeline': pipeline_name,
            'unit_id': idx,
            'isnad_text': isnad,
            'has_isnad': unit.get('has_isnad', False),
            'char_end': unit.get('char_end', 0),
        })
    return boundaries


def get_boundary_context(text: str, position: int, context_len: int = 50) -> tuple[str, str, str]:
    """
    Get text context around a boundary position.

    Returns (before, at_position, after)
    """
    start = max(0, position - context_len)
    end = min(len(text), position + context_len)

    before = text[start:position]
    at_pos = text[position:min(position + 30, len(text))]
    after = text[position:end]

    return before, at_pos, after


def create_html_visualization(
    text: str,
    deepseek_units: list[dict],
    camelbert_units: list[dict],
    baseline_units: list[dict],
    output_path: str
) -> None:
    """
    Create interactive HTML visualization of boundary markers.
    """
    import html
    text_escaped = html.escape(text)

    # Extract boundaries from each pipeline
    deepseek_boundaries = extract_boundaries(deepseek_units, 'deepseek')
    camelbert_boundaries = extract_boundaries(camelbert_units, 'camelbert')
    baseline_boundaries = extract_boundaries(baseline_units, 'baseline')

    all_boundaries = (deepseek_boundaries + camelbert_boundaries + baseline_boundaries)
    all_boundaries.sort(key=lambda x: x['position'])

    # Create position index: position -> list of boundaries at that position
    position_boundaries = defaultdict(list)
    for boundary in all_boundaries:
        position_boundaries[boundary['position']].append(boundary)

    # Calculate statistics
    deepseek_positions = set(b['position'] for b in deepseek_boundaries)
    camelbert_positions = set(b['position'] for b in camelbert_boundaries)
    baseline_positions = set(b['position'] for b in baseline_boundaries)

    overlap_all_3 = deepseek_positions & camelbert_positions & baseline_positions
    overlap_ds_cb = (deepseek_positions & camelbert_positions) - baseline_positions
    overlap_ds_bl = (deepseek_positions & baseline_positions) - camelbert_positions
    overlap_cb_bl = (camelbert_positions & baseline_positions) - deepseek_positions

    # Build segment list with boundary markers inserted
    segments = []
    boundary_positions = sorted(position_boundaries.keys())
    boundary_idx = 0

    for char_idx in range(len(text)):
        if boundary_idx < len(boundary_positions) and char_idx == boundary_positions[boundary_idx]:
            boundaries_here = position_boundaries[boundary_positions[boundary_idx]]
            segments.append({
                'type': 'boundary',
                'position': char_idx,
                'boundaries': boundaries_here,
            })
            boundary_idx += 1

        # Add text character
        segments.append({
            'type': 'text',
            'char': text[char_idx],
            'position': char_idx,
        })

    # Prepare JSON data for JavaScript
    boundaries_json = json.dumps(all_boundaries)
    segments_json = json.dumps(segments)

    # Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Isnad Boundary Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}

        h1 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }}

        .subtitle {{
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            border-left: 4px solid;
        }}

        .stat-card.deepseek {{
            border-color: #2ecc71;
        }}

        .stat-card.camelbert {{
            border-color: #e74c3c;
        }}

        .stat-card.baseline {{
            border-color: #3498db;
        }}

        .stat-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}

        .stat-value {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }}

        .controls {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}

        .toggle-btn {{
            padding: 10px 15px;
            border: 2px solid;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
            font-size: 14px;
        }}

        .toggle-btn.deepseek {{
            border-color: #2ecc71;
            color: #2ecc71;
        }}

        .toggle-btn.deepseek.active {{
            background: #2ecc71;
            color: white;
        }}

        .toggle-btn.camelbert {{
            border-color: #e74c3c;
            color: #e74c3c;
        }}

        .toggle-btn.camelbert.active {{
            background: #e74c3c;
            color: white;
        }}

        .toggle-btn.baseline {{
            border-color: #3498db;
            color: #3498db;
        }}

        .toggle-btn.baseline.active {{
            background: #3498db;
            color: white;
        }}

        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 6px;
            flex-wrap: wrap;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .legend-marker {{
            width: 3px;
            height: 20px;
            border-radius: 2px;
        }}

        .legend-marker.deepseek {{ background: #2ecc71; }}
        .legend-marker.camelbert {{ background: #e74c3c; }}
        .legend-marker.baseline {{ background: #3498db; }}

        .overlap-info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
        }}

        .overlap-row {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            flex-wrap: wrap;
        }}

        .overlap-stat {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .overlap-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        .text-container {{
            background: #fafafa;
            padding: 25px;
            border-radius: 6px;
            line-height: 2;
            direction: rtl;
            text-align: right;
            word-break: break-word;
            white-space: pre-wrap;
            font-family: 'Arabic Typesetting', 'Arial Unicode MS', Arial, sans-serif;
            font-size: 15px;
            color: #333;
            position: relative;
            border: 1px solid #e0e0e0;
        }}

        .boundary-marker {{
            position: relative;
            display: inline;
            margin: 0 -2px;
            padding: 0 2px;
            border-radius: 1px;
        }}

        .boundary-marker .marker-line {{
            display: inline-block;
            width: 2px;
            height: 1.5em;
            margin: 0 -1px;
            vertical-align: middle;
            opacity: 0.7;
        }}

        .boundary-marker .tooltip {{
            display: none;
            position: absolute;
            bottom: 120%;
            left: -50px;
            background: #2c3e50;
            color: white;
            padding: 10px 12px;
            border-radius: 4px;
            white-space: nowrap;
            font-size: 11px;
            z-index: 1000;
            margin-bottom: 8px;
            pointer-events: none;
            white-space: normal;
            width: 150px;
            text-align: center;
        }}

        .boundary-marker:hover .tooltip {{
            display: block;
        }}

        .boundary-marker.hidden {{
            opacity: 0;
            pointer-events: none;
        }}

        .boundary-marker.hidden .marker-line {{
            display: none;
        }}

        .match-indicator {{
            display: inline-block;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            margin: 0 1px;
            vertical-align: middle;
        }}

        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
            line-height: 1.6;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 15px;
            }}
            .text-container {{
                font-size: 13px;
                padding: 15px;
            }}
            .controls {{
                flex-direction: column;
            }}
            .toggle-btn {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Isnad Boundary Visualization</h1>
        <p class="subtitle">Compare boundary detection across three pipelines: Deepseek (semantic), CAMeL-BERT (token classification), and Baseline (linguistic rules)</p>

        <div class="stats-grid">
            <div class="stat-card deepseek">
                <div class="stat-label">Deepseek Boundaries</div>
                <div class="stat-value">{len(deepseek_boundaries)}</div>
            </div>
            <div class="stat-card camelbert">
                <div class="stat-label">CAMeL-BERT Boundaries</div>
                <div class="stat-value">{len(camelbert_boundaries)}</div>
            </div>
            <div class="stat-card baseline">
                <div class="stat-label">Baseline Boundaries</div>
                <div class="stat-value">{len(baseline_boundaries)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Text Length</div>
                <div class="stat-value">{len(text):,} chars</div>
            </div>
        </div>

        <div class="overlap-info">
            <strong>Boundary Overlap Analysis (±80 char tolerance):</strong>
            <div class="overlap-row">
                <div class="overlap-stat">
                    <div class="overlap-color" style="background: #16a085;"></div>
                    <span>All 3 pipelines: <strong>{len(overlap_all_3)}</strong></span>
                </div>
                <div class="overlap-stat">
                    <div class="overlap-color" style="background: #d68910;"></div>
                    <span>Deepseek + CAMeL: <strong>{len(overlap_ds_cb)}</strong></span>
                </div>
                <div class="overlap-stat">
                    <div class="overlap-color" style="background: #8e44ad;"></div>
                    <span>Deepseek + Baseline: <strong>{len(overlap_ds_bl)}</strong></span>
                </div>
                <div class="overlap-stat">
                    <div class="overlap-color" style="background: #c0392b;"></div>
                    <span>CAMeL + Baseline: <strong>{len(overlap_cb_bl)}</strong></span>
                </div>
            </div>
        </div>

        <div class="controls">
            <button class="toggle-btn deepseek active" onclick="togglePipeline('deepseek')">
                Deepseek (Green)
            </button>
            <button class="toggle-btn camelbert active" onclick="togglePipeline('camelbert')">
                CAMeL-BERT (Red)
            </button>
            <button class="toggle-btn baseline active" onclick="togglePipeline('baseline')">
                Baseline (Blue)
            </button>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-marker deepseek"></div>
                <span><strong>Deepseek:</strong> Semantic unit boundaries (LLM)</span>
            </div>
            <div class="legend-item">
                <div class="legend-marker camelbert"></div>
                <span><strong>CAMeL-BERT:</strong> Token isnad boundaries (ML)</span>
            </div>
            <div class="legend-item">
                <div class="legend-marker baseline"></div>
                <span><strong>Baseline:</strong> Linguistic markers (Rule-based)</span>
            </div>
        </div>

        <div class="text-container" id="text-container"></div>

        <div class="footer">
            <p><strong>How to read:</strong> Each colored vertical line marks where a pipeline detects a narrative unit boundary. Hover over any marker to see the text context and isnad information. Use toggle buttons to show/hide specific pipelines.</p>
            <p><strong>Interpretation:</strong> Where all three pipelines align (same position), there's strong agreement. Where they diverge, it reflects their different design goals (semantic grouping vs. exhaustive boundary detection).</p>
        </div>
    </div>

    <script>
        const allBoundaries = {boundaries_json};
        const allSegments = {segments_json};
        const textContent = `{text_escaped}`;

        function togglePipeline(pipeline) {{
            const btn = event.target;
            btn.classList.toggle('active');

            const markers = document.querySelectorAll(`.boundary-marker[data-pipeline="${{pipeline}}"]`);
            markers.forEach(marker => {{
                if (btn.classList.contains('active')) {{
                    marker.classList.remove('hidden');
                }} else {{
                    marker.classList.add('hidden');
                }}
            }});
        }}

        function renderText() {{
            const container = document.getElementById('text-container');
            container.innerHTML = '';

            // Group boundaries by position for easier lookup
            const boundaryMap = {{}};
            allBoundaries.forEach(b => {{
                if (!boundaryMap[b.position]) {{
                    boundaryMap[b.position] = [];
                }}
                boundaryMap[b.position].push(b);
            }});

            // Build text with boundary markers
            const sortedPositions = Object.keys(boundaryMap)
                .map(p => parseInt(p))
                .sort((a, b) => a - b);

            let lastPos = 0;

            for (let pos of sortedPositions) {{
                // Add text before this boundary
                if (pos > lastPos) {{
                    const textSpan = document.createElement('span');
                    textSpan.textContent = textContent.substring(lastPos, pos);
                    container.appendChild(textSpan);
                }}

                // Add boundary markers
                const boundariesHere = boundaryMap[pos];
                const uniquePipelines = new Set(boundariesHere.map(b => b.pipeline));

                boundariesHere.forEach(boundary => {{
                    const marker = document.createElement('span');
                    marker.className = `boundary-marker`;
                    marker.setAttribute('data-pipeline', boundary.pipeline);
                    marker.setAttribute('data-position', pos);

                    // Get context text
                    const contextStart = Math.max(0, pos - 30);
                    const contextEnd = Math.min(textContent.length, pos + 30);
                    const before = textContent.substring(contextStart, pos);
                    const after = textContent.substring(pos, contextEnd);

                    // Truncate if too long
                    const displayBefore = before.length > 25 ? '...' + before.slice(-25) : before;
                    const displayAfter = after.length > 25 ? after.slice(0, 25) + '...' : after;

                    const tooltipText = `${{boundary.pipeline.toUpperCase()}}\\nChar ${{pos}}\\nBefore: "${{displayBefore}}"\\nAfter: "${{displayAfter}}"`;

                    const lineSpan = document.createElement('span');
                    lineSpan.className = 'marker-line';
                    lineSpan.style.backgroundColor = getPipelineColor(boundary.pipeline);
                    marker.appendChild(lineSpan);

                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip';
                    tooltip.innerHTML = tooltipText.replace(/\\n/g, '<br>');
                    marker.appendChild(tooltip);

                    container.appendChild(marker);
                }});

                lastPos = pos;
            }}

            // Add remaining text
            if (lastPos < textContent.length) {{
                const textSpan = document.createElement('span');
                textSpan.textContent = textContent.substring(lastPos);
                container.appendChild(textSpan);
            }}
        }}

        function getPipelineColor(pipeline) {{
            switch(pipeline) {{
                case 'deepseek': return '#2ecc71';
                case 'camelbert': return '#e74c3c';
                case 'baseline': return '#3498db';
                default: return '#95a5a6';
            }}
        }}

        // Initialize
        renderText();
    </script>
</body>
</html>
"""

    # Save HTML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"Saved visualization to {output_path}")
    logger.info(f"File size: {len(html_content) / 1024 / 1024:.2f} MB")
    logger.info(f"Boundaries: Deepseek={len(deepseek_boundaries)}, CAMeL-BERT={len(camelbert_boundaries)}, Baseline={len(baseline_boundaries)}")


def main():
    parser = argparse.ArgumentParser(
        description='Create interactive isnad boundary visualization'
    )
    parser.add_argument(
        '--text',
        required=True,
        help='Path to cleaned text'
    )
    parser.add_argument(
        '--deepseek',
        required=True,
        help='Path to gold_standard_deepseek.json'
    )
    parser.add_argument(
        '--camelbert',
        required=True,
        help='Path to camelbert_narrative_units.json'
    )
    parser.add_argument(
        '--baseline',
        required=True,
        help='Path to baseline_v4_narrative_units.json'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save HTML visualization'
    )

    args = parser.parse_args()

    try:
        logger.info("Loading text and units...")
        text = load_text(args.text)
        deepseek_units = load_units(args.deepseek)
        camelbert_units = load_units(args.camelbert)
        baseline_units = load_units(args.baseline)

        logger.info(f"Text: {len(text)} chars")
        logger.info(f"Deepseek: {len(deepseek_units)} units")
        logger.info(f"CAMeL-BERT: {len(camelbert_units)} units")
        logger.info(f"Baseline: {len(baseline_units)} units")

        logger.info("Creating visualization...")
        create_html_visualization(
            text,
            deepseek_units,
            camelbert_units,
            baseline_units,
            args.output
        )
        logger.info("Visualization completed successfully")

    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        raise


if __name__ == '__main__':
    main()
