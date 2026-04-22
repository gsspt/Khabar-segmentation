#!/usr/bin/env python3
"""
Flexible boundary token conversion for any text/raw_inference pair.

Usage:
    python scripts/convert_boundary_tokens_flexible.py \\
      --raw_inference results/alDarrab/camelbert_alDarrab_raw_inference.json \\
      --corpus data/processed/alDarrab_clean.txt \\
      --output results/camelbert_alDarrab_char_boundaries.json
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert CAMeL-BERT raw inference to char-level boundaries"
    )
    parser.add_argument(
        "--raw_inference",
        required=True,
        help="Path to raw inference JSON",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to corpus text file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save char boundaries JSON",
    )

    args = parser.parse_args()

    # Load files
    print("Chargement des fichiers...")
    with open(args.raw_inference, encoding='utf-8') as f:
        raw = json.load(f)["inference_results"]

    corpus_text = Path(args.corpus).read_text(encoding="utf-8")
    N = len(corpus_text)

    print(f"  Corpus        : {N:,} chars")
    print(f"  Total tokens  : {raw['total_tokens']:,}")

    # ── Étape 0 : Gérer les chunks (convertir offsets relatifs en absolus) ─────

    SPECIAL = {"[PAD]", "[CLS]", "[SEP]", "[UNK]"}

    tokens = raw["tokens"]
    offsets = raw["offsets"]
    preds = raw["predictions"]
    probs = raw["probabilities"]

    # Find chunk boundaries (SEP tokens mark end of chunks)
    sep_indices = [i for i, t in enumerate(tokens) if t == "[SEP]"]
    has_chunks = len(sep_indices) > 1

    print(f"  Chunks détectés: {len(sep_indices)}")

    # Calculate chunk start positions in corpus
    # Chunks are 500 chars with stride=500 (no overlap) or minimal overlap
    # Determine stride from metadata if available
    metadata = json.load(open(args.raw_inference, encoding='utf-8')).get('metadata', {})
    chunk_size = metadata.get('chunk_size', 500)

    # Use stride=500 (full chunk size, minimal overlap)
    # This means each chunk processes the next 500 chars starting from previous chunk end
    stride = chunk_size  # 500 chars

    chunk_starts = [i * stride for i in range(len(sep_indices))]

    print(f"  Chunk size: {chunk_size} chars")
    print(f"  Stride: {stride} chars")
    print(f"  Chunk starts (in corpus): {chunk_starts[:5]}...")

    # ── Étape 1 : extraction directe par char_start (avec conversion chunk→corpus) ──

    # Pour chaque char_start unique, garder la proba max
    char_to_prob = {}
    char_to_pred = {}
    char_to_tok = {}
    char_to_end = {}

    for tok_idx, (tok, off, pred, prob) in enumerate(zip(tokens, offsets, preds, probs)):
        if tok in SPECIAL:
            continue
        cs, ce = off
        if cs == 0 and ce == 0:
            continue

        # Convert to absolute corpus position
        if has_chunks:
            # Find which chunk this token belongs to
            chunk_id = 0
            for sep_idx in sep_indices:
                if tok_idx > sep_idx:
                    chunk_id += 1

            if chunk_id < len(chunk_starts):
                chunk_offset = chunk_starts[chunk_id]
                cs_abs = cs + chunk_offset
                ce_abs = ce + chunk_offset
            else:
                cs_abs = cs
                ce_abs = ce
        else:
            cs_abs = cs
            ce_abs = ce

        if cs_abs not in char_to_prob or prob > char_to_prob[cs_abs]:
            char_to_prob[cs_abs] = prob
            char_to_pred[cs_abs] = pred
            char_to_tok[cs_abs] = tok
            char_to_end[cs_abs] = ce_abs

    total_unique = len(char_to_prob)
    boundary_chars_all = sorted(
        cs for cs, pred in char_to_pred.items() if pred == 1
    )
    n_boundary = len(boundary_chars_all)

    print(f"  Positions uniques        : {total_unique:,}")
    print(f"  Boundary tokens (pred=1) : {n_boundary:,}")
    print(f"  Taux boundary            : {n_boundary/total_unique*100:.1f}%")

    # Premiers boundary tokens pour validation
    print(f"\n  Premiers boundary tokens : {len(boundary_chars_all[:10])} samples")
    for cs in boundary_chars_all[:10]:
        prob = char_to_prob[cs]
        print(f"    char={cs:6d}  prob={prob:.4f}")

    # ── Étape 2 : regrouper en clusters de boundary tokens ──────────────────────

    print("\nRegroupement des boundary tokens en clusters (GAP_CLUSTER = 20 chars)...")
    GAP_CLUSTER = 20
    clusters = []
    if boundary_chars_all:
        cur_start = boundary_chars_all[0]
        cur_end = char_to_end[boundary_chars_all[0]]
        cur_toks = [boundary_chars_all[0]]

        for cs in boundary_chars_all[1:]:
            if cs - cur_end <= GAP_CLUSTER:
                cur_end = max(cur_end, char_to_end[cs])
                cur_toks.append(cs)
            else:
                clusters.append(
                    {
                        "char_start": cur_start,
                        "char_end": cur_end,
                        "n_tokens": len(cur_toks),
                        "tokens": [char_to_tok[c] for c in cur_toks[:10]],
                        "text_context": corpus_text[
                            max(0, cur_start - 10) : cur_end + 60
                        ],
                    }
                )
                cur_start = cs
                cur_end = char_to_end[cs]
                cur_toks = [cs]
        clusters.append(
            {
                "char_start": cur_start,
                "char_end": cur_end,
                "n_tokens": len(cur_toks),
                "tokens": [char_to_tok[c] for c in cur_toks[:10]],
                "text_context": corpus_text[max(0, cur_start - 10) : cur_end + 60],
            }
        )

    print(f"  Clusters isnad : {len(clusters)}")

    # Longueurs moyennes
    if clusters:
        avg_cluster_len = sum(c["char_end"] - c["char_start"] for c in clusters) / len(
            clusters
        )
        print(f"  Longueur moy. cluster : {avg_cluster_len:.0f} chars")

    # ── Étape 3 : extraire khabar boundaries ────────────────────────────────────

    khabar_boundaries = [c["char_start"] for c in clusters]

    # ── Étape 4 : analyse de la distribution des gaps ────────────────────────────

    print("\nDistribution des gaps entre clusters (= longueur des khabars) :")
    gaps = []
    for i in range(1, len(clusters)):
        gap = clusters[i]["char_start"] - clusters[i - 1]["char_end"]
        gaps.append(gap)

    if gaps:
        gaps_sorted = sorted(gaps)
        print(f"  Min gap : {min(gaps):,} chars")
        print(f"  Mediane : {gaps_sorted[len(gaps_sorted)//2]:,} chars")
        print(f"  Max gap : {max(gaps):,} chars")
        print(f"  Gaps > 100 chars : {sum(1 for g in gaps if g > 100)}")
        print(f"  Gaps > 500 chars : {sum(1 for g in gaps if g > 500)}")

    # ── Étape 5 : sauvegarde ──────────────────────────────────────────────────────

    output = {
        "metadata": {
            "method": "direct_char_extraction_from_raw_inference",
            "description": (
                "Extraction directe des boundary tokens (pred=1) via les offsets "
                "globaux du raw_inference. Deduplication par char_start (proba max). "
                "Clustering en segments isnad (GAP_CLUSTER=20 chars). "
                "Debut de chaque cluster = frontiere khabar candidate."
            ),
            "source_raw_inference": args.raw_inference,
            "corpus": args.corpus,
            "corpus_chars": N,
            "unique_char_positions": total_unique,
            "boundary_tokens_count": n_boundary,
            "gap_cluster_max": GAP_CLUSTER,
            "n_clusters_isnad": len(clusters),
            "n_khabar_boundaries": len(khabar_boundaries),
        },
        "khabar_boundaries": [
            {
                "boundary_id": i,
                "char_start": c["char_start"],
                "char_end": c["char_end"],
                "n_tokens": c["n_tokens"],
                "tokens": c["tokens"],
                "text_context": c["text_context"],
            }
            for i, c in enumerate(clusters)
        ],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Sauvegarde : {out_path}")
    print(f"  {len(clusters)} clusters isnad -> {len(khabar_boundaries)} khabar boundaries")


if __name__ == "__main__":
    main()
