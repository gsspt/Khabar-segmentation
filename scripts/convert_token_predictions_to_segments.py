#!/usr/bin/env python3
"""
Convert token-level CAMeL-BERT predictions to khabar-level segments.

The model predicts token-level boundaries (isnad=1, non-isnad=0) with 99.93% accuracy.
This script converts those token predictions back to actual text segments.

Core challenge: Map tokens → character positions → segment boundaries
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer


def tokenizer_offsets(text: str, tokenizer) -> List[Tuple[int, int, str]]:
    """
    Get character-level offsets for each token.

    Returns: List of (start_char, end_char, token_text)
    """
    # Tokenize with offset mapping
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors=None
    )

    offsets = encoded['offset_mapping']
    token_ids = encoded['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    result = []
    for token, (start, end) in zip(tokens, offsets):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        if start == end:  # Skip special tokens with no offset
            continue
        token_text = text[start:end] if start < len(text) else ''
        result.append((start, end, token))

    return result


def cluster_boundary_tokens(
    boundary_indices: List[int],
    token_offsets: List[Tuple[int, int, str]],
    max_gap: int = 5
) -> List[Tuple[int, int]]:
    """
    Cluster adjacent boundary tokens into segment boundaries.

    Adjacent tokens (distance < max_gap) mark the same segment boundary.
    Returns list of (start_char, end_char) for each cluster.
    """
    if not boundary_indices:
        return []

    clusters = []
    current_cluster_start = boundary_indices[0]
    current_cluster_end = current_cluster_start

    for i in range(1, len(boundary_indices)):
        idx = boundary_indices[i]
        prev_idx = boundary_indices[i - 1]

        # Check gap between consecutive boundary tokens
        if idx - prev_idx <= max_gap:
            # Continue current cluster
            current_cluster_end = idx
        else:
            # End current cluster, start new one
            if current_cluster_start < len(token_offsets):
                start_char = token_offsets[current_cluster_start][0]
                end_char = token_offsets[current_cluster_end][1]
                clusters.append((start_char, end_char))

            current_cluster_start = idx
            current_cluster_end = idx

    # Add final cluster
    if current_cluster_start < len(token_offsets):
        start_char = token_offsets[current_cluster_start][0]
        end_char = token_offsets[current_cluster_end][1]
        clusters.append((start_char, end_char))

    return clusters


def extract_segments(
    text: str,
    boundary_positions: List[Tuple[int, int]]
) -> List[Dict]:
    """
    Extract text segments based on boundary positions.

    Segments are text between boundaries.
    """
    if not boundary_positions:
        return [{'text': text.strip(), 'start': 0, 'end': len(text), 'type': 'unknown'}]

    # Sort boundaries by start position
    boundaries = sorted(boundary_positions)

    segments = []
    current_pos = 0

    for boundary_start, boundary_end in boundaries:
        # Text before this boundary
        if current_pos < boundary_start:
            seg_text = text[current_pos:boundary_start].strip()
            if seg_text:
                segments.append({
                    'text': seg_text,
                    'start': current_pos,
                    'end': boundary_start,
                    'type': 'non-isnad'
                })

        # The boundary itself (likely an isnad)
        seg_text = text[boundary_start:boundary_end].strip()
        if seg_text:
            segments.append({
                'text': seg_text,
                'start': boundary_start,
                'end': boundary_end,
                'type': 'isnad'
            })

        current_pos = boundary_end

    # Remaining text
    if current_pos < len(text):
        seg_text = text[current_pos:].strip()
        if seg_text:
            segments.append({
                'text': seg_text,
                'start': current_pos,
                'end': len(text),
                'type': 'non-isnad'
            })

    return segments


def convert_predictions_to_segments(
    text: str,
    predictions: List[int],
    tokenizer,
    threshold: float = 0.5,
    max_cluster_gap: int = 5
) -> Dict:
    """
    Convert token-level predictions to khabar-level segments.

    Args:
        text: Original document text
        predictions: Token-level predictions (0/1 array)
        tokenizer: CAMeL-BERT tokenizer
        threshold: Confidence threshold (if using probabilities)
        max_cluster_gap: Max token distance to cluster as same boundary

    Returns:
        Dict with segments and metadata
    """
    # Step 1: Get token offsets
    token_offsets = tokenizer_offsets(text, tokenizer)

    # Step 2: Find boundary token indices
    boundary_indices = np.where(np.array(predictions) == 1)[0].tolist()

    # Step 3: Cluster adjacent boundary tokens
    boundary_clusters = cluster_boundary_tokens(
        boundary_indices,
        token_offsets,
        max_gap=max_cluster_gap
    )

    # Step 4: Extract segments
    segments = extract_segments(text, boundary_clusters)

    return {
        'text_length': len(text),
        'total_tokens': len(predictions),
        'boundary_tokens': len(boundary_indices),
        'boundary_clusters': len(boundary_clusters),
        'segments': segments,
        'total_segments': len(segments)
    }


def main():
    """Test the conversion on a sample."""

    # Load a test example
    corpus_file = Path('data/processed/binary_classification_dataset/binary_classification_examples.jsonl')

    if not corpus_file.exists():
        print(f"ERROR: {corpus_file} not found")
        return

    # Load tokenizer
    model_path = Path('checkpoints/camelbert_binary_classification_final')
    if not model_path.exists():
        print(f"ERROR: Model {model_path} not found")
        return

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Load example
    import json
    with open(corpus_file) as f:
        examples = [json.loads(line) for line in f]

    example = examples[10]  # Pick a sample

    # Reconstruct text
    text = ' [SEP] '.join([seg['text'] for seg in example['segments']])

    print(f"Example {example['akhbar_id']}")
    print(f"Segments: {example['segment_types']}")
    print(f"Text: {text[:100]}...\n")

    # TODO: Run actual model inference to get predictions
    # For now, create dummy predictions for demonstration
    predictions = np.random.randint(0, 2, size=512).tolist()

    # Convert to segments
    result = convert_predictions_to_segments(text, predictions, tokenizer)

    print(f"[RESULTS]")
    print(f"  Total segments detected: {result['total_segments']}")
    print(f"  Boundary clusters: {result['boundary_clusters']}")
    print(f"\nFirst 3 segments:")
    for i, seg in enumerate(result['segments'][:3]):
        print(f"  {i+1}. [{seg['type']}] {seg['text'][:50]}...")


if __name__ == '__main__':
    main()
