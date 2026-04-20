#!/usr/bin/env python3
"""
Complete pipeline: CAMeL-BERT token prediction → khabar segments

Steps:
1. Load fine-tuned CAMeL-BERT model
2. Run inference on document with token-to-character offset mapping
3. Convert token-level predictions to character-level boundaries
4. Extract and return actual khabar segments
5. Evaluate against gold standard
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForTokenClassification


class CamelBertSegmentationPipeline:
    """End-to-end pipeline for CAMeL-BERT segmentation."""

    def __init__(self, model_path: str = 'checkpoints/camelbert_binary_classification_final'):
        """Initialize model and tokenizer."""
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"[INFO] Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        print(f"[OK] Model loaded")

    def infer_boundaries(self, text: str, max_length: int = 512) -> Dict:
        """
        Run inference and get token-level predictions with offsets.

        Returns:
            Dict with predictions and offset mapping
        """
        # Tokenize with offset mapping
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )

        # Get predictions
        with torch.no_grad():
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}

            outputs = self.model(**{k: v for k, v in encoded.items() if k != 'offset_mapping'})
            logits = outputs.logits[0]  # [seq_len, 2]

        # Convert to predictions
        preds = np.argmax(logits.cpu().numpy(), axis=-1)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]  # prob of class 1

        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        offsets = encoded['offset_mapping'][0].numpy()

        return {
            'text': text,
            'tokens': tokens,
            'predictions': preds,
            'probabilities': probs,
            'offsets': offsets,
        }

    def cluster_boundaries(
        self,
        predictions: np.ndarray,
        offsets: np.ndarray,
        max_token_gap: int = 3
    ) -> List[Tuple[int, int]]:
        """
        Cluster adjacent boundary tokens into segment boundaries.

        Adjacent tokens (distance < max_token_gap) mark the same boundary.
        Returns list of (char_start, char_end) for each boundary cluster.
        """
        boundary_indices = np.where(predictions == 1)[0]

        if len(boundary_indices) == 0:
            return []

        clusters = []
        current_start_idx = boundary_indices[0]
        current_end_idx = boundary_indices[0]

        for i in range(1, len(boundary_indices)):
            idx = boundary_indices[i]
            prev_idx = boundary_indices[i - 1]

            if idx - prev_idx <= max_token_gap:
                # Continue cluster
                current_end_idx = idx
            else:
                # End current cluster
                char_start = offsets[current_start_idx][0]
                char_end = offsets[current_end_idx][1]
                clusters.append((char_start, char_end))

                current_start_idx = idx
                current_end_idx = idx

        # Final cluster
        char_start = offsets[current_start_idx][0]
        char_end = offsets[current_end_idx][1]
        clusters.append((char_start, char_end))

        return clusters

    def extract_segments(
        self,
        text: str,
        boundary_clusters: List[Tuple[int, int]]
    ) -> List[Dict]:
        """
        Extract segments from text given boundary positions.

        Each boundary is treated as the start of an isnad.
        Non-boundary text between boundaries is narrative.
        """
        if not boundary_clusters:
            return [{
                'text': text.strip(),
                'start': 0,
                'end': len(text),
                'type': 'unknown'
            }]

        segments = []
        current_pos = 0

        for bound_start, bound_end in sorted(boundary_clusters):
            # Text before boundary (narrative)
            if current_pos < bound_start:
                seg_text = text[current_pos:bound_start].strip()
                if seg_text:
                    segments.append({
                        'text': seg_text,
                        'start': current_pos,
                        'end': bound_start,
                        'type': 'prose'
                    })

            # Boundary itself (isnad)
            seg_text = text[bound_start:bound_end].strip()
            if seg_text:
                segments.append({
                    'text': seg_text,
                    'start': bound_start,
                    'end': bound_end,
                    'type': 'isnad'
                })

            current_pos = bound_end

        # Remaining text
        if current_pos < len(text):
            seg_text = text[current_pos:].strip()
            if seg_text:
                segments.append({
                    'text': seg_text,
                    'start': current_pos,
                    'end': len(text),
                    'type': 'prose'
                })

        return segments

    def segment(self, text: str) -> Dict:
        """
        Full pipeline: text → segments

        Returns:
            Dict with segments and statistics
        """
        # Step 1: Infer boundaries
        result = self.infer_boundaries(text)

        # Step 2: Cluster boundary tokens
        boundary_clusters = self.cluster_boundaries(
            result['predictions'],
            result['offsets']
        )

        # Step 3: Extract segments
        segments = self.extract_segments(text, boundary_clusters)

        return {
            'segments': segments,
            'total_segments': len(segments),
            'total_boundary_tokens': np.sum(result['predictions']),
            'boundary_clusters': len(boundary_clusters),
            'mean_boundary_prob': float(np.mean(result['probabilities'][result['predictions'] == 1])) if np.sum(result['predictions']) > 0 else 0,
        }


def main():
    """Test the pipeline."""
    import sys

    # Initialize pipeline
    try:
        pipeline = CamelBertSegmentationPipeline()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load test text
    corpus_file = Path('data/processed/kitab_uqala_reference_corpus.txt')
    if not corpus_file.exists():
        print(f"ERROR: {corpus_file} not found")
        sys.exit(1)

    with open(corpus_file, encoding='utf-8') as f:
        text = f.read()

    print(f"[INFO] Text size: {len(text)} chars")

    # Run segmentation
    print(f"[INFO] Running segmentation...")
    result = pipeline.segment(text[:50000])  # Start with first 50K chars for testing

    print(f"\n[RESULTS]")
    print(f"  Total segments: {result['total_segments']}")
    print(f"  Boundary tokens: {result['total_boundary_tokens']}")
    print(f"  Boundary clusters: {result['boundary_clusters']}")
    print(f"  Mean boundary prob: {result['mean_boundary_prob']:.4f}")

    print(f"\nFirst 5 segments:")
    for i, seg in enumerate(result['segments'][:5]):
        print(f"  {i+1}. [{seg['type']}] {seg['text'][:60]}...")


if __name__ == '__main__':
    main()
