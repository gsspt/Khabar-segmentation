#!/usr/bin/env python3
"""
Visualize CAMeL-BERT detected boundaries in interactive HTML.

Usage:
    python scripts/visualize_boundaries.py \
      --boundaries results/[TEXT]/camelbert_[TEXT]_char_boundaries.json \
      --corpus data/processed/[TEXT]_clean.txt \
      --output results/visualization_[TEXT]_boundaries.html
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def generate_html(boundaries: List[Dict], corpus_text: str, text_name: str) -> str:
    """Generate interactive HTML visualization with background highlighting."""

    # Build text with background-highlighted boundaries (no inline markers)
    html_parts = []
    last_end = 0

    for i, boundary in enumerate(boundaries):
        start = boundary["char_start"]
        end = boundary["char_end"]
        max_prob = boundary.get("max_prob", 0)

        # Text before boundary
        if last_end < start:
            before = corpus_text[last_end:start]
            html_parts.append(f"<span class='text'>{escape_html(before)}</span>")

        # Boundary region with background highlight (no inline marker)
        prob_color = get_color_for_prob(max_prob)
        boundary_text = corpus_text[start:end]
        tokens_str = ', '.join(boundary.get('tokens', [])[:5])

        # Get context around boundary
        context_start = max(0, start - 100)
        context_end = min(len(corpus_text), end + 100)
        context = corpus_text[context_start:context_end]
        context_escaped = escape_html(context).replace('<br>\n', ' | ')

        tooltip = f"Boundary {i}: prob={max_prob:.3f}\\nTokens: {tokens_str}\\nContext: ...{context_escaped}..."

        # Use background highlight with subtle border and opacity
        html_parts.append(
            f"<span class='boundary-region' "
            f"style='background-color:{prob_color}; opacity: 0.3; "
            f"border-bottom: 2px solid {prob_color};' "
            f"title='{tooltip}' data-boundary='{i}'>"
            f"{escape_html(boundary_text)}</span>"
        )

        last_end = end

    # Remaining text
    if last_end < len(corpus_text):
        remaining = corpus_text[last_end:]
        html_parts.append(f"<span class='text'>{escape_html(remaining)}</span>")

    highlighted_text = "".join(html_parts)

    # Statistics
    stats = {
        "total_boundaries": len(boundaries),
        "corpus_length": len(corpus_text),
        "avg_segment_length": len(corpus_text) / len(boundaries) if boundaries else 0,
        "text_coverage": sum(b["char_end"] - b["char_start"] for b in boundaries) / len(corpus_text) * 100,
        "high_confidence": sum(1 for b in boundaries if b.get("max_prob", 0) > 0.95),
        "medium_confidence": sum(1 for b in boundaries if 0.8 <= b.get("max_prob", 0) <= 0.95),
        "low_confidence": sum(1 for b in boundaries if b.get("max_prob", 0) < 0.8),
    }

    # Prepare JSON data for JavaScript
    boundaries_json = json.dumps(boundaries)
    corpus_json = json.dumps(corpus_text)

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CAMeL-BERT Boundaries: {text_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            direction: rtl;
            text-align: right;
        }}
        .header {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat {{
            background: #f9f9f9;
            padding: 10px;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            font-weight: 600;
        }}
        .stat-value {{
            font-size: 1.5em;
            color: #333;
            font-weight: bold;
            margin-top: 5px;
        }}
        .text-viewer {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            line-height: 1.8;
            font-size: 16px;
            word-wrap: break-word;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'Traditional Arabic', 'Arial Unicode MS', sans-serif;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .text {{
            color: #333;
        }}
        .boundary-region {{
            cursor: pointer;
            transition: opacity 0.2s ease;
            padding: 2px 0;
            border-radius: 2px;
        }}
        .boundary-region:hover {{
            opacity: 0.6 !important;
            box-shadow: 0 0 3px rgba(0,0,0,0.3);
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border: 1px solid #ccc;
        }}
        .confidence-info {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .confidence-info h3 {{
            margin-top: 0;
            color: #333;
        }}
        .confidence-bar {{
            width: 100%;
            height: 30px;
            background: linear-gradient(to right, #ff6b6b, #ffd93d, #6bcf7f);
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            margin: 10px 0;
        }}
        @media (prefers-color-scheme: dark) {{
            body {{
                background: #1e1e1e;
                color: #e0e0e0;
            }}
            .header, .text-viewer, .confidence-info {{
                background: #2d2d2d;
                color: #e0e0e0;
            }}
            .stat {{
                background: #3d3d3d;
                border-left-color: #2196F3;
            }}
            .text {{
                color: #e0e0e0;
            }}
            .header h1 {{
                color: #e0e0e0;
            }}
            .stat-label, .stat-value {{
                color: #e0e0e0;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>CAMeL-BERT Boundary Detection: {text_name}</h1>
        <p>Interactive visualization of detected narrative unit boundaries</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-label">Total Boundaries</div>
                <div class="stat-value">{stats['total_boundaries']}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Avg Segment Length</div>
                <div class="stat-value">{stats['avg_segment_length']:.0f} chars</div>
            </div>
            <div class="stat">
                <div class="stat-label">Text Coverage</div>
                <div class="stat-value">{stats['text_coverage']:.1f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">High Confidence</div>
                <div class="stat-value">{stats['high_confidence']}</div>
            </div>
        </div>
    </div>

    <div class="legend">
        <div style="margin-bottom: 10px; font-size: 0.9em; color: #666;">
            <strong>Background Colors:</strong> Boundaries are highlighted by background color based on confidence
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #6bcf7f; opacity: 0.3;"></div>
            <span>High confidence (>0.95)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ffd93d; opacity: 0.3;"></div>
            <span>Medium confidence (0.8-0.95)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ff6b6b; opacity: 0.3;"></div>
            <span>Low confidence (<0.8)</span>
        </div>
        <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            <strong>Interaction:</strong> Hover over highlighted regions to see tooltips. Click to see full context details.
        </div>
    </div>

    <div class="text-viewer">
        {highlighted_text}
    </div>

    <div class="confidence-info">
        <h3>Confidence Distribution</h3>
        <div>
            <strong>High confidence (>0.95):</strong> {stats['high_confidence']} boundaries ({100*stats['high_confidence']/max(1,stats['total_boundaries']):.1f}%)
            <div class="confidence-bar" style="width: {100*stats['high_confidence']/max(1,stats['total_boundaries'])}%; background-color: #6bcf7f;"></div>
        </div>
        <div style="margin-top: 15px;">
            <strong>Medium confidence (0.8-0.95):</strong> {stats['medium_confidence']} boundaries ({100*stats['medium_confidence']/max(1,stats['total_boundaries']):.1f}%)
            <div class="confidence-bar" style="width: {100*stats['medium_confidence']/max(1,stats['total_boundaries'])}%; background-color: #ffd93d;"></div>
        </div>
        <div style="margin-top: 15px;">
            <strong>Low confidence (<0.8):</strong> {stats['low_confidence']} boundaries ({100*stats['low_confidence']/max(1,stats['total_boundaries']):.1f}%)
            <div class="confidence-bar" style="width: {100*stats['low_confidence']/max(1,stats['total_boundaries'])}%; background-color: #ff6b6b;"></div>
        </div>
    </div>

    <div id="boundary-detail" class="boundary-detail" style="display: none; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-top: 20px;">
        <h3>Boundary Details</h3>
        <p><strong>Boundary ID:</strong> <span id="detail-id">-</span></p>
        <p><strong>Confidence:</strong> <span id="detail-prob">-</span></p>
        <p><strong>Tokens:</strong> <span id="detail-tokens">-</span></p>
        <p><strong>Full Context (±150 chars):</strong></p>
        <pre id="detail-context" style="background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;"></pre>
    </div>

    <script>
        // Store boundary data for detail view
        const boundaries = {boundaries_json};
        const corpus = {corpus_json};

        // Add click handlers to boundary regions
        document.querySelectorAll('.boundary-region').forEach(el => {{
            el.addEventListener('click', function(e) {{
                const bid = parseInt(this.getAttribute('data-boundary'));
                const b = boundaries[bid];
                const start = b.char_start;
                const end = b.char_end;

                // Remove previous selection highlight
                document.querySelectorAll('.boundary-region').forEach(e => {{
                    e.style.opacity = '0.3';
                }});

                // Highlight selected boundary
                this.style.opacity = '0.7';

                // Show detail
                document.getElementById('detail-id').textContent = bid;
                document.getElementById('detail-prob').textContent = (b.max_prob ? b.max_prob.toFixed(4) : 'N/A');
                document.getElementById('detail-tokens').textContent = b.tokens.join(', ');

                // Full context
                const ctx_start = Math.max(0, start - 150);
                const ctx_end = Math.min(corpus.length, end + 150);
                const ctx = corpus.substring(ctx_start, ctx_end);
                document.getElementById('detail-context').textContent = ctx;

                document.getElementById('boundary-detail').style.display = 'block';
                document.getElementById('boundary-detail').scrollIntoView({{behavior: 'smooth'}});
            }});
        }});
    </script>
</body>
</html>
"""

    return html


def escape_html(text: str) -> str:
    """Escape HTML special characters and preserve newlines."""
    # First escape special characters
    text = (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))
    # Then convert newlines to <br> tags
    text = text.replace("\n", "<br>\n")
    return text


def get_color_for_prob(prob: float) -> str:
    """Get color based on probability confidence."""
    if prob > 0.95:
        return "#6bcf7f"  # Green
    elif prob > 0.8:
        return "#ffd93d"  # Yellow
    else:
        return "#ff6b6b"  # Red


def main():
    parser = argparse.ArgumentParser(description="Visualize CAMeL-BERT boundaries")
    parser.add_argument("--boundaries", required=True, help="Path to char_boundaries JSON")
    parser.add_argument("--corpus", required=True, help="Path to corpus text file")
    parser.add_argument("--output", required=True, help="Path to save HTML output")

    args = parser.parse_args()

    logger.info("Loading files...")
    with open(args.boundaries, encoding="utf-8") as f:
        boundaries_data = json.load(f)

    corpus_text = Path(args.corpus).read_text(encoding="utf-8")

    boundaries = boundaries_data.get("khabar_boundaries", [])
    text_name = Path(args.boundaries).stem.replace("camelbert_", "").replace("_char_boundaries", "")

    logger.info(f"  Boundaries: {len(boundaries)}")
    logger.info(f"  Corpus: {len(corpus_text):,} chars")

    logger.info("\nGenerating HTML...")
    html = generate_html(boundaries, corpus_text, text_name)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"✓ Saved to {out_path}")
    logger.info(f"  Open in browser to view interactive visualization")


if __name__ == "__main__":
    main()
