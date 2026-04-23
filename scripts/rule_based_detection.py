#!/usr/bin/env python3
"""
Flexible Baseline v4 runner — segmentez any text without requiring a gold standard.

Usage:
    python scripts/baseline_flexible.py \\
      --input data/processed/alDarrab_clean.txt \\
      --output results/baseline_v4_alDarrab_segments.json
"""

import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


# ── Normalisation ─────────────────────────────────────────────────────────────

def norm(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # diacritiques
    for ch in 'أإآٱ':
        text = text.replace(ch, 'ا')
    text = text.replace('ة', 'ه')
    text = text.replace('ى', 'ي')
    return text


# ── Lexique des verbes de transmission ────────────────────────────────────────

CHAIN_VERB_STARTS: Tuple[str, ...] = (
    'حدثن', 'حدثي', 'حدثه', 'حدثك',
    'اخبرن', 'اخبري', 'اخبره',
    'سمعت', 'سمعن', 'سمعه',
    'انشدن', 'انشدي',
    'انبان', 'انباي',
    'حكين', 'حكيت',
    'بلغن', 'بلغي',
    'رواه', 'روين', 'روان',
    'كتبن', 'كتبي',
    'يقول', 'يروي',
    'ذكرن', 'ذكري',
    'وحدث', 'واخبر', 'وسمعت', 'وسمعن',
    'وانشد', 'وانبا', 'وحكي', 'وبلغ',
    'وروى', 'ورواه',
)


# ── Détection des débuts d'isnad ───────────────────────────────────────────────

def find_isnad_starts(text: str) -> List[Tuple[int, str]]:
    """Trouve tous les débuts d'isnad (verbes de transmission)."""
    starts = []
    for verb in CHAIN_VERB_STARTS:
        for m in re.finditer(verb, text):
            starts.append((m.start(), verb))

    # Dédupliquer et trier
    starts.sort()
    last = -1
    deduped = []
    for pos, verb in starts:
        if pos > last:
            deduped.append((pos, verb))
            last = pos + len(verb)
    return deduped


# ── Détection de fin d'isnad ───────────────────────────────────────────────────

def find_isnad_end(text: str, istart: int, window: int = 500) -> int:
    """
    Détecte la fin de l'isnad.
    Stratégie : Chercher le premier قال suivi d'un NON-verbe de transmission.
    """
    search_until = min(istart + window, len(text))
    subset = text[istart:search_until]

    for m in re.finditer(r'قال', subset):
        qala_pos = istart + m.end()
        # Sauter espaces
        ws_end = qala_pos
        while ws_end < len(text) and text[ws_end] in ' \n\t':
            ws_end += 1

        # Vérifier si le mot suivant est un verbe de chaîne
        is_chain_verb = any(
            text[ws_end:ws_end+len(v)] == v for v in CHAIN_VERB_STARTS
        )

        if not is_chain_verb:
            return ws_end

    return istart + 150


# ── Segmentation ───────────────────────────────────────────────────────────────

def segment(text: str) -> List[Dict]:
    """Segmente le texte en akhbars (isnad + khabar)."""
    t = norm(text)
    starts = find_isnad_starts(t)
    if not starts:
        return []

    akhbars: List[Dict] = []
    covered = 0

    for i, (istart, verb) in enumerate(starts):
        if istart < covered:
            continue

        # Borne de la prochaine unité
        next_start = starts[i + 1][0] if i + 1 < len(starts) else len(t)

        # Fin de l'isnad
        iend = find_isnad_end(t, istart, window=min(900, next_start - istart + 50))

        # L'isnad ne doit pas dépasser le prochain marqueur
        if iend <= istart or iend > next_start:
            iend = min(istart + 400, next_start)

        isnad_len = iend - istart
        khabar_len = next_start - iend

        # Filtre de sanité minimal : isnad trop court = faux positif
        if isnad_len < 8:
            continue

        isnad_text = text[istart:iend].strip()
        khabar_text = text[iend:next_start].strip()

        if not isnad_text:
            continue

        akhbars.append({
            'id': len(akhbars),
            'start': istart,
            'end': iend,
            'isnad_text': isnad_text[:100],  # premiers 100 chars
            'isnad_len': isnad_len,
            'khabar_len': khabar_len,
        })

        covered = iend

    return akhbars


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baseline v4 flexible runner for any Arabic text"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input text file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save boundaries JSON",
    )

    args = parser.parse_args()

    # Load text
    print(f"Loading text from {args.input}...")
    with open(args.input, encoding='utf-8') as f:
        text = f.read()

    print(f"  Text size: {len(text):,} chars")

    # Segment
    print("\nSegmenting with Baseline v4...")
    akhbars = segment(text)
    print(f"  Boundaries detected: {len(akhbars)}")

    if akhbars:
        gaps = []
        for i in range(1, len(akhbars)):
            gap = akhbars[i]['start'] - akhbars[i-1]['end']
            gaps.append(gap)

        if gaps:
            gaps_sorted = sorted(gaps)
            print(f"\n  Gap statistics:")
            print(f"    Min: {min(gaps):,} chars")
            print(f"    Median: {gaps_sorted[len(gaps_sorted)//2]:,} chars")
            print(f"    Max: {max(gaps):,} chars")
            print(f"    Gaps > 100: {sum(1 for g in gaps if g > 100)}")
            print(f"    Gaps > 500: {sum(1 for g in gaps if g > 500)}")

    # Save
    output = {
        "metadata": {
            "method": "baseline_v4_linguistic",
            "description": "Isnad detection via transmission verbs (حدثنا، أخبرنا، قال)",
            "source_text": args.input,
            "text_size_chars": len(text),
            "boundaries_count": len(akhbars),
        },
        "segments": akhbars,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Saved to {out_path}")


if __name__ == "__main__":
    main()
