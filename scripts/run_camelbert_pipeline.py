#!/usr/bin/env python3
"""
Unified CAMeL-BERT Pipeline: Clean → Infer → Convert → Visualize

Runs the complete segmentation pipeline in one command:
    python scripts/run_camelbert_pipeline.py --input data/raw/TEXT.raw.txt

Usage:
    python scripts/run_camelbert_pipeline.py \
      --input data/raw/TEXT.raw.txt \
      [--model models/camelbert_binary_classification_final] \
      [--text_name TEXT] \
      [--gap_cluster 20] \
      [--force] [--skip_clean] [--skip_inference] [--skip_convert] [--skip_viz]
"""

import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ===== LOGGING SETUP (BEFORE local imports) =====
def setup_logging() -> logging.Logger:
    """Configure root logger and return a named logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('camelbert_pipeline')


logger = setup_logging()

# ===== SYS.PATH SETUP (BEFORE local imports) =====
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ===== LOCAL IMPORTS (after logging and sys.path are ready) =====
try:
    from clean_openiti_text import clean_openiti_text
    from convert_boundary_tokens_direct import (
        extract_boundaries,
        cluster_boundaries,
        validate_boundaries,
    )
    from visualize_boundaries import generate_html
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    sys.exit(1)


# ===== HELPER FUNCTIONS =====

def derive_text_name(input_path: Path) -> str:
    """Extract text identifier from filename (e.g., 'text.raw.txt' → 'text')."""
    return input_path.name.split('.')[0]


def build_output_paths(project_root: Path, text_name: str) -> Dict[str, Path]:
    """Build all output paths for a given text."""
    return {
        'clean': project_root / 'data' / 'processed' / f'{text_name}_clean.txt',
        'raw_inference': project_root / 'results' / text_name / f'camelbert_{text_name}_raw_inference.json',
        'boundaries': project_root / 'results' / text_name / f'camelbert_{text_name}_char_boundaries.json',
        'visualization': project_root / 'results' / text_name / f'visualization_{text_name}_camelbert_boundaries.html',
    }


def load_model(model_path: Path) -> Tuple[Any, Any]:
    """Load CAMeL-BERT model and tokenizer with GPU support."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please download and place it at models/camelbert_binary_classification_final/"
        )

    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        )

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForTokenClassification.from_pretrained(str(model_path))
    model.eval()

    # GPU detection and placement
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            model = model.cuda()
            logger.info(f"Using GPU: {device_name}")
        else:
            logger.info("Using CPU (no GPU detected)")
    except ImportError:
        pass  # torch will be imported again in run_inference()

    return tokenizer, model


def run_inference(
    text: str,
    model: Any,
    tokenizer: Any,
    model_path: Path,
    text_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Dict[str, Any]:
    """
    Run CAMeL-BERT inference on full text using chunking with overlap.

    Converts chunk-relative offsets to absolute corpus positions.
    Handles overlap deduplication by skipping first 5 tokens of subsequent chunks.
    """
    import numpy as np

    try:
        import torch
        from scipy.special import softmax
    except ImportError as e:
        raise ImportError(
            f"Required libraries missing: {e}. "
            "Install with: pip install torch scipy numpy"
        )

    logger.info(f"Running CAMeL-BERT inference on {len(text):,} chars...")

    # Device detection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Estimate chunks
    num_chunks = (len(text) + chunk_size - 1) // chunk_size
    logger.info(f"  ~{num_chunks} chunks at {chunk_size} chars/chunk with {overlap} char overlap")

    # Accumulators
    all_tokens = []
    all_predictions = []
    all_offsets = []
    all_probabilities = []
    chunks_processed = 0

    start_time = time.time()

    # Chunking loop
    for start_char in range(0, len(text), chunk_size):
        end_char = min(start_char + chunk_size + overlap, len(text))
        chunk_text = text[start_char:end_char]

        # Tokenize chunk
        encoded = tokenizer(
            chunk_text,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
        )

        # Move to device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]  # [seq_len, num_labels]

        # Predictions and probabilities
        preds = np.argmax(logits.cpu().numpy(), axis=-1)
        probs = softmax(logits.cpu().numpy(), axis=-1)
        boundary_probs = probs[:, 1].tolist()  # class 1 = boundary

        # Tokens and offsets
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        offsets_chunk_relative = encoded['offset_mapping'][0].numpy()

        # Convert chunk-relative offsets to absolute corpus positions
        offsets_absolute = [
            [int(start_char + int(offset[0])), int(start_char + int(offset[1]))]
            for offset in offsets_chunk_relative
        ]

        # Deduplication: skip first 5 tokens of each chunk after the first
        if chunks_processed > 0 and len(all_tokens) > 0:
            skip = 5
            preds = preds[skip:]
            tokens = tokens[skip:]
            offsets_absolute = offsets_absolute[skip:]
            boundary_probs = boundary_probs[skip:]

        # Accumulate
        all_tokens.extend(tokens)
        all_predictions.extend(preds.tolist())
        all_offsets.extend(offsets_absolute)
        all_probabilities.extend(boundary_probs)

        chunks_processed += 1

        # Progress every 10 chunks
        if chunks_processed % 10 == 0:
            logger.debug(f"  Processed {chunks_processed}/{num_chunks} chunks...")

    elapsed_secs = time.time() - start_time
    elapsed_str = f"{elapsed_secs / 60:.1f} min" if elapsed_secs >= 60 else f"{elapsed_secs:.1f}s"

    # Validate lengths match
    assert len(all_tokens) == len(all_predictions) == len(all_offsets) == len(all_probabilities), \
        f"Length mismatch: tokens={len(all_tokens)}, preds={len(all_predictions)}, " \
        f"offsets={len(all_offsets)}, probs={len(all_probabilities)}"

    boundary_count = sum(all_predictions)
    boundary_pct = 100 * boundary_count / len(all_tokens) if all_tokens else 0

    logger.info(
        f"  Done in {elapsed_str} — {len(all_tokens):,} tokens, "
        f"{boundary_count:,} boundary tokens ({boundary_pct:.1f}%)"
    )

    # Assemble output dict
    result = {
        'metadata': {
            'corpus': f'{text_name}_clean.txt',
            'corpus_size_chars': int(len(text)),
            'total_tokens': int(len(all_tokens)),
            'model': model_path.name,
            'chunk_size': int(chunk_size),
            'overlap': int(overlap),
            'chunks_processed': int(chunks_processed),
            'boundary_tokens_count': int(boundary_count),
            'boundary_percentage': float(round(boundary_pct, 2)),
            'offset_format': 'ABSOLUTE corpus character positions [char_start, char_end]',
        },
        'inference_results': {
            'total_tokens': int(len(all_tokens)),
            'tokens': all_tokens,
            'offsets': all_offsets,
            'predictions': [int(p) for p in all_predictions],
            'probabilities': [float(p) for p in all_probabilities],
        },
    }

    return result


# ===== STEP FUNCTIONS =====

def step_clean(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    logger: logging.Logger,
) -> str:
    """Step 1: Clean raw OpenITI text."""
    output_path = paths['clean']

    # Check if already exists
    if output_path.exists() and not args.force:
        logger.info(f"[Step 1/4] Skipping clean (output exists): {output_path}")
        return output_path.read_text(encoding='utf-8')

    if args.skip_clean:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--skip_clean was set but {output_path} does not exist. "
                "Run without --skip_clean first."
            )
        logger.info(f"[Step 1/4] Skipping clean (flag set), reading {output_path}")
        return output_path.read_text(encoding='utf-8')

    # Run cleaning
    logger.info(f"[Step 1/4] Cleaning raw text...")
    logger.info(f"  Input:  {args.input}")
    logger.info(f"  Output: {output_path}")

    start_time = time.time()
    stats = clean_openiti_text(str(args.input), str(output_path))
    elapsed = time.time() - start_time

    logger.info(
        f"[Step 1/4] Done in {elapsed:.1f}s — "
        f"{stats.get('cleaned_chars', 0):,} chars, "
        f"{stats.get('words', 0):,} words"
    )

    return output_path.read_text(encoding='utf-8')


def step_inference(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    corpus_text: str,
    text_name: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Step 2: Run CAMeL-BERT inference."""
    output_path = paths['raw_inference']

    # Check if already exists
    if output_path.exists() and not args.force:
        logger.info(f"[Step 2/4] Skipping inference (output exists): {output_path}")
        with open(output_path, encoding='utf-8') as f:
            return json.load(f)

    if args.skip_inference:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--skip_inference was set but {output_path} does not exist. "
                "Run without --skip_inference first."
            )
        logger.info(f"[Step 2/4] Skipping inference (flag set), reading {output_path}")
        with open(output_path, encoding='utf-8') as f:
            return json.load(f)

    # Run inference
    logger.info(f"[Step 2/4] Running CAMeL-BERT inference...")

    tokenizer, model = load_model(args.model)
    start_time = time.time()

    raw_inference = run_inference(
        corpus_text,
        model,
        tokenizer,
        args.model,
        text_name,
        chunk_size=500,
        overlap=50,
    )

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(raw_inference, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time

    logger.info(
        f"[Step 2/4] Saved to {output_path} "
        f"({output_path.stat().st_size / (1024*1024):.1f} MB)"
    )

    return raw_inference


def step_convert(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    raw_inference: Dict[str, Any],
    logger: logging.Logger,
) -> List[Dict]:
    """Step 3: Post-process boundaries."""
    output_path = paths['boundaries']

    # Check if already exists
    if output_path.exists() and not args.force:
        logger.info(f"[Step 3/4] Skipping convert (output exists): {output_path}")
        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)
            return data.get('khabar_boundaries', [])

    if args.skip_convert:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--skip_convert was set but {output_path} does not exist. "
                "Run without --skip_convert first."
            )
        logger.info(f"[Step 3/4] Skipping convert (flag set), reading {output_path}")
        with open(output_path, encoding='utf-8') as f:
            data = json.load(f)
            return data.get('khabar_boundaries', [])

    # Extract and post-process
    logger.info(f"[Step 3/4] Post-processing boundaries (gap_cluster={args.gap_cluster})...")

    start_time = time.time()
    corpus_text = paths['clean'].read_text(encoding='utf-8')

    results = raw_inference['inference_results']
    tokens = results['tokens']
    offsets = results['offsets']
    predictions = results['predictions']
    probabilities = results['probabilities']

    # Extract boundaries
    char_to_info = extract_boundaries(
        tokens, offsets, predictions, probabilities, corpus_text
    )

    # Cluster boundaries
    clusters = cluster_boundaries(
        char_to_info, corpus_text, gap_cluster=args.gap_cluster
    )

    # Validate
    validation = validate_boundaries(clusters, corpus_text)

    elapsed = time.time() - start_time

    logger.info(
        f"[Step 3/4] Done in {elapsed:.1f}s — "
        f"{len(clusters)} clusters found"
    )

    # Assemble output
    output_data = {
        'metadata': {
            'method': 'direct_global_offsets',
            'description': (
                'Direct extraction using global HuggingFace offset_mapping. '
                'Each token offset is already a global corpus character position. '
                'No chunk localization needed.'
            ),
            'model': raw_inference['metadata']['model'],
            'corpus_chars': len(corpus_text),
            'total_tokens': len(tokens),
            'boundary_tokens': sum(predictions),
            'unique_positions': len(char_to_info),
            'gap_cluster': args.gap_cluster,
            'clusters_found': len(clusters),
            'validation': validation,
        },
        'khabar_boundaries': clusters,
    }

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"[Step 3/4] Saved to {output_path}")

    return clusters


def step_visualize(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    text_name: str,
    logger: logging.Logger,
) -> None:
    """Step 4: Generate HTML visualization."""
    output_path = paths['visualization']

    # Check if already exists
    if output_path.exists() and not args.force:
        logger.info(f"[Step 4/4] Skipping visualization (output exists): {output_path}")
        return

    if args.skip_viz:
        logger.info(f"[Step 4/4] Skipping visualization (flag set)")
        return

    # Generate visualization
    logger.info(f"[Step 4/4] Generating HTML visualization...")

    start_time = time.time()

    # Load boundaries and corpus
    with open(paths['boundaries'], encoding='utf-8') as f:
        boundaries_data = json.load(f)

    corpus_text = paths['clean'].read_text(encoding='utf-8')
    boundaries = boundaries_data['khabar_boundaries']

    # Generate HTML
    html = generate_html(boundaries, corpus_text, text_name)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    elapsed = time.time() - start_time
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"[Step 4/4] Done in {elapsed:.1f}s — "
        f"saved to {output_path} ({file_size_mb:.1f} MB)"
    )


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(
        description='Unified CAMeL-BERT khabar segmentation pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full pipeline on new text
  python scripts/run_camelbert_pipeline.py --input data/raw/text.raw.txt

  # With custom model location
  python scripts/run_camelbert_pipeline.py --input data/raw/text.raw.txt \\
    --model /path/to/model

  # Resume after inference failed (skip expensive steps)
  python scripts/run_camelbert_pipeline.py --input data/raw/text.raw.txt \\
    --skip_clean --skip_inference

  # Re-run visualization only
  python scripts/run_camelbert_pipeline.py --input data/raw/text.raw.txt \\
    --skip_clean --skip_inference --skip_convert
        '''
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to raw OpenITI text file',
    )
    parser.add_argument(
        '--text_name',
        type=str,
        default=None,
        help='Override text identifier (default: derived from filename)',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=Path('models/camelbert_binary_classification_final'),
        help='Path to CAMeL-BERT model directory',
    )
    parser.add_argument(
        '--gap_cluster',
        type=int,
        default=20,
        help='Gap threshold for boundary clustering in chars (optimal: 20)',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-run all steps even if outputs already exist',
    )
    parser.add_argument(
        '--skip_clean',
        action='store_true',
        help='Skip step 1 (text cleaning)',
    )
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='Skip step 2 (CAMeL-BERT inference)',
    )
    parser.add_argument(
        '--skip_convert',
        action='store_true',
        help='Skip step 3 (boundary post-processing)',
    )
    parser.add_argument(
        '--skip_viz',
        action='store_true',
        help='Skip step 4 (HTML visualization)',
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Derive paths
    project_root = Path(__file__).resolve().parent.parent
    text_name = args.text_name or derive_text_name(args.input)
    paths = build_output_paths(project_root, text_name)

    logger.info(f"=" * 70)
    logger.info(f"CAMeL-BERT Segmentation Pipeline")
    logger.info(f"=" * 70)
    logger.info(f"Text: {text_name}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Gap cluster: {args.gap_cluster}")
    logger.info(f"")

    pipeline_start = time.time()

    try:
        # Step 1: Clean
        corpus_text = step_clean(args, paths, logger)

        # Step 2: Inference
        raw_inference = step_inference(args, paths, corpus_text, text_name, logger)

        # Step 3: Convert
        clusters = step_convert(args, paths, raw_inference, logger)

        # Step 4: Visualize
        step_visualize(args, paths, text_name, logger)

        # Summary
        pipeline_elapsed = time.time() - pipeline_start
        elapsed_str = f"{pipeline_elapsed / 60:.1f} min" if pipeline_elapsed >= 60 else f"{pipeline_elapsed:.1f}s"

        logger.info(f"")
        logger.info(f"Pipeline complete in {elapsed_str}")
        logger.info(f"  {paths['clean']}")
        logger.info(f"  {paths['raw_inference']}")
        logger.info(f"  {paths['boundaries']}")
        logger.info(f"  {paths['visualization']}")
        logger.info(f"=" * 70)

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
