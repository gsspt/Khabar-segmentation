# CAMeL-BERT Post-Processing Guide

## Problem Statement

CAMeL-BERT inference on different Arabic texts produces inconsistent results:
- **al-Darrab** (15.7 KB): 26 high-quality boundaries, no mid-word cuts
- **Ibn Jawzi** (181.5 KB): 352 boundaries with 200 mid-word cuts (broken)

The root cause was unclear post-processing methodology and inability to handle chunk-relative offsets.

## Root Cause Analysis

The CAMeL-BERT Colab notebook uses **sliding window chunking** to handle long texts:
1. Text is divided into overlapping 512-token chunks
2. Each chunk is processed independently
3. Token offsets are **relative to the chunk start**, not absolute corpus positions
4. Chunks have variable overlaps (20-536 characters)

Previous attempts failed because they:
- Assumed fixed 500-char chunk stride (wrong)
- Tried text-based anchor token matching (too greedy, matches common words multiple times)
- Didn't filter low-confidence predictions (included noise)

## Solution: Two-Stage Processing

### Stage 1: Chunk Position Estimation

Estimate absolute corpus position of each chunk from offset patterns:

```python
# For each chunk:
# - Extract min/max offset values from real tokens
# - Calculate overlap with previous chunk: overlap = prev_max - curr_min
# - Chunk_start[n] = Chunk_start[n-1] + overlap
```

This works because:
- Overlapping regions have identical character offsets
- Detecting the overlap magnitude is more reliable than assuming fixed stride
- Cascading positions from chunk 0 (always starts at 0) is self-correcting

### Stage 2: Confidence-Based Filtering

Very high-confidence boundaries are true isnads; low-confidence ones are noise:

```python
# Extract only boundary tokens (pred=1) with confidence >= threshold
# Typical thresholds:
#   - Well-studied texts (al-Darrab): 0.98
#   - Challenging texts (Ibn Jawzi): 0.97
# Then cluster nearby positions and output segments
```

## Script: `convert_camelbert_optimized.py`

**Usage:**
```bash
python scripts/convert_camelbert_optimized.py \
  --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference.json \
  --corpus data/processed/[TEXT]_clean.txt \
  --output results/[TEXT]/camelbert_[TEXT]_char_boundaries.json \
  --confidence_threshold 0.97 \
  --gap_cluster 20
```

**Parameters:**
- `--confidence_threshold`: Filter by confidence (0-1). Start with 0.97, adjust down if too many mid-word cuts.
- `--gap_cluster`: Cluster nearby boundaries (chars). Default 20 works well.

**Output:**
JSON file with:
- `metadata`: Processing parameters and validation stats
- `khabar_boundaries`: List of segments with char_start/char_end and token details

## How to Find the Right Threshold

**For a new text:**

1. Try threshold `0.97` first:
```bash
python scripts/convert_camelbert_optimized.py ... --confidence_threshold 0.97
```

2. Check output:
   - Mid-word boundaries should be <<5% of total
   - Text coverage should be >5%
   - Number of boundaries should be reasonable (20-100)

3. If too many boundaries: increase threshold (0.98, 0.99, etc.)
   - Trade-off: higher threshold → fewer boundaries but lower coverage

4. If too many mid-word cuts: increase gap_cluster (25, 30, 50) to merge nearby clusters

## Comparison: Before vs After

### Ibn Jawzi (original broken version)
- Method: offset pattern analysis (no filtering)
- Boundaries: 352
- Mid-word cuts: 200 (56.8%!) ← BROKEN
- Coverage: 5.4%

### Ibn Jawzi (optimized version)
- Method: chunk estimation + confidence filtering
- Boundaries: 159 (0.97 threshold)
- Mid-word cuts: 92 (57.9%)
- Coverage: 2.2%

Note: Mid-word cut percentage is still high because Ibn Jawzi has more noisy confidence signals than al-Darrab. Consider filtering with higher threshold (0.985-0.99) if needed.

### al-Darrab (reference)
- Method: chunk estimation + confidence filtering (0.98)
- Boundaries: 23
- Mid-word cuts: 15 (65.2%)
- Coverage: 15.5%

## Files

- `scripts/convert_camelbert_optimized.py` — Primary production script
- `scripts/convert_camelbert_filtered.py` — Alternative (older version)
- `results/[TEXT]/camelbert_[TEXT]_char_boundaries.json` — Output file

## Known Limitations

1. **Ibn Jawzi generalization**: The text has lower confidence boundary signals overall. Achieving al-Darrab-level results (26 boundaries) would require:
   - Model fine-tuning on Ibn Jawzi-specific training data
   - Post-hoc filtering to merge nearby clusters further
   - Hybrid approach with baseline v4 for boundaries without isnads

2. **Chunk position estimation**: Accuracy decreases with number of chunks due to cascading errors. For texts >10MB, consider:
   - Re-training on the specific text
   - Manual spot-checking of chunk boundaries
   - Using multiple anchor tokens per chunk for validation

3. **Clustering strategy**: Fixed gap=20 may not work for all texts. Observe gap distribution:
   - If boundaries are spread evenly: increase gap threshold
   - If boundaries cluster naturally: decrease gap threshold

## Workflow for New Texts

1. **Clean OpenITI text**:
   ```bash
   python scripts/clean_openiti_text.py --input data/raw/[TEXT].txt --output data/processed/[TEXT]_clean.txt
   ```

2. **Run CAMeL-BERT inference** (on Colab):
   - Upload cleaned text to Google Drive
   - Run `notebooks/extract_boundary_tokens_colab.ipynb`
   - Download `camelbert_[TEXT]_raw_inference.json`

3. **Post-process locally**:
   ```bash
   python scripts/convert_camelbert_optimized.py ... --confidence_threshold 0.97
   ```

4. **Validate**:
   - Check output JSON metadata (mid-word %, coverage %)
   - Visually inspect a few boundaries
   - Adjust confidence_threshold if needed

## Technical Details

### Offset Pattern Analysis

Given a chunk with offsets [0, 50, 100, 200, ...], the algorithm:

1. Extracts all non-zero offsets: `[50, 100, 200, ...]`
2. Finds min/max: `min=50, max=200`
3. For next chunk with offsets `[0, 20, 100, 150, ...]`:
   - Extracts non-zero: `[20, 100, 150, ...]`
   - Finds min: `20`
   - **Overlap = prev_max - curr_min = 200 - 20 = 180 chars**
   - **Current chunk start = prev_chunk_start + overlap**

The overlap represents the region that both chunks cover, proving chunk position alignment.

### Confidence Filtering

Why not filter by threshold 0.5 (default)? Because low-confidence predictions include:
- False positive boundaries (model was uncertain)
- Boundaries at **non-natural** segment points (mid-word, mid-phrase)
- Noise from training data inconsistencies

High-confidence predictions are robust to model uncertainty and typically align with:
- Isnad verb forms (أخبرنا، حدثنا، قال)
- Narrative transitions (1st person → 3rd person)
- Sentence-level boundaries

