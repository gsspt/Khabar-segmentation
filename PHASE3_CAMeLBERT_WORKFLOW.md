# Phase 3: CAMeL-BERT Inference Workflow for alDarrab

**Updated**: 2026-04-22  
**Notebook**: `notebooks/extract_boundary_tokens_final.ipynb` (enhanced)  
**Output Format**: Raw inference with tokens, offsets, predictions, probabilities

---

## Overview

The inference workflow has 3 main steps:

```
1. COLAB INFERENCE          (Google Colab notebook)
   ↓
   results/camelbert_alDarrab_raw_inference.json
   ↓
2. LOCAL POST-PROCESSING    (convert_boundary_tokens_direct.py)
   ↓
   results/camelbert_alDarrab_char_boundaries.json
   ↓
3. COMPARISON               (compare_pipelines.py)
   ↓
   results/comparison_alDarrab.json
```

---

## Step 1: Run CAMeL-BERT Inference in Colab

### Notebook Changes Made

The `extract_boundary_tokens_final.ipynb` has been updated with:

| Cell | Change | Purpose |
|------|--------|---------|
| **Cell 4** | `corpus_file = Path('data/processed/alDarrab_clean.txt')` | Load cleaned alDarrab text |
| **Cell 5** | Added `scipy.special.softmax` import | Compute boundary probabilities |
| **Cell 5** | Added `all_probabilities` list | Capture P(token is boundary) |
| **Cell 8** | Save `inference_results` dict with tokens, offsets, predictions, probabilities | Enhanced raw format |
| **Cell 9** | New validation cell | Verify output format |

### File Structure: Raw Inference JSON

```json
{
  "metadata": {
    "corpus": "alDarrab_clean.txt",
    "corpus_size_chars": 27638,
    "total_tokens": 8345,
    "boundary_tokens_count": 1680,
    "boundary_percentage": 20.14,
    "model": "camelbert_binary_classification_final"
  },
  "inference_results": {
    "total_tokens": 8345,
    "tokens": ["#", "بسم", "الله", ...],
    "offsets": [[0, 1], [2, 6], [7, 11], ...],
    "predictions": [0, 0, 0, 1, 0, ...],
    "probabilities": [0.05, 0.08, 0.12, 0.92, 0.15, ...]
  }
}
```

**Key Fields**:
- `tokens`: Tokenized words from the text
- `offsets`: Character positions [[start, end], ...] for each token in original text
- `predictions`: Binary prediction (0=not boundary, 1=boundary)
- `probabilities`: Confidence score for boundary prediction (0.0-1.0)

### Colab Workflow

1. **Upload cleaned text to Google Drive**
   ```
   → Upload data/processed/alDarrab_clean.txt to Drive
   ```

2. **Open notebook in Colab**
   ```
   https://colab.research.google.com/
   → File → Open → Select extract_boundary_tokens_final.ipynb
   ```

3. **Run cells sequentially**
   - Cell 1-3: Setup (mount Drive, install dependencies, load model)
   - Cell 4: Load alDarrab corpus (UPDATED)
   - Cell 5: Run inference with overlap (UPDATED)
   - Cell 8: Save raw inference JSON (UPDATED)
   - Cell 9: Validate format (NEW)

4. **Expected Output**
   ```
   Processing complete
     Chunks processed: 14
     Total tokens: 8,345
     Boundary tokens: 1,680
     Percentage: 20.14%
   
   Saved: results/camelbert_alDarrab_raw_inference.json
   Size: 4.2 MB
   ```

5. **Download results**
   ```
   → Download results/camelbert_alDarrab_raw_inference.json
   → Save to local: results/
   ```

---

## Step 2: Local Post-Processing

After downloading raw inference to local machine:

```bash
python scripts/convert_boundary_tokens_direct.py \
  --input results/camelbert_alDarrab_raw_inference.json \
  --corpus data/processed/alDarrab_clean.txt \
  --output results/camelbert_alDarrab_char_boundaries.json
```

### What This Does

1. **Extract boundary tokens** (pred=1 from raw inference)
2. **Deduplicate by char_start** (keep max probability per position)
3. **Cluster with gap=20 chars** (group contiguous isnad tokens)
4. **Extract cluster starts** as khabar boundaries
5. **Evaluate vs gold standard** (if available)

### Expected Output

```json
{
  "metadata": {
    "method": "direct_char_extraction_from_raw_inference",
    "n_khabar_boundaries": 185,
    "evaluation_tol80": {
      "f1": 0.864,
      "precision": 0.924,
      "recall": 0.812
    }
  },
  "khabar_boundaries": [
    {
      "boundary_id": 0,
      "char_start": 24,
      "char_end": 120,
      "n_tokens": 12,
      "text_context": "# أخبرنا أبو محمد الحسن..."
    },
    ...
  ]
}
```

---

## Step 3: Run Baseline v4 Comparison

Simultaneously, run baseline on the same text:

```bash
python scripts/baseline_v4.py \
  --input data/processed/alDarrab_clean.txt \
  --output results/baseline_v4_alDarrab_segments.json
```

---

## Step 4: Compare Results

Once you have both outputs:

```bash
python scripts/compare_pipelines.py \
  --camelbert results/camelbert_alDarrab_char_boundaries.json \
  --baseline results/baseline_v4_alDarrab_segments.json \
  --output results/comparison_alDarrab.json
```

---

## Key Advantages of Enhanced Format

✓ **Full traceability**: Every token has offset, prediction, probability  
✓ **Confidence filtering**: Can filter by probability threshold if needed  
✓ **Error analysis**: Can identify which tokens have low confidence  
✓ **Reproducibility**: Raw predictions saved, post-processing is deterministic  
✓ **Clustering flexibility**: Can rerun clustering with different gaps without re-inferring  

---

## Timeline Estimate

| Step | Time | Notes |
|------|------|-------|
| **Colab Inference** | 2-5 min | Depends on alDarrab size (~27 KB) |
| **Download** | 1 min | ~4 MB file |
| **Post-processing** | 30 sec | Local machine |
| **Baseline v4** | 10 sec | Fast linguistic approach |
| **Comparison** | 1 min | Generate metrics |
| **Total** | ~10 min | Mostly Colab time |

---

## Quality Checks

After each step, verify:

1. **Raw Inference**:
   - All lists have same length: `len(tokens) == len(offsets) == len(predictions) == len(probabilities)`
   - Offsets are valid: `0 <= start <= end <= corpus_length`
   - Probabilities in [0, 1]: `min(probs) >= 0.0 and max(probs) <= 1.0`

2. **Char Boundaries**:
   - Boundaries are in order: `char_start[i] < char_start[i+1]`
   - Within corpus: `0 <= char_start <= corpus_length`

3. **Comparison**:
   - F1 score in [0, 1]
   - Precision, Recall in [0, 1]
   - TP + FP + FN = Total predictions

---

## Next: Multi-Pipeline Comparison

Once all 3 pipelines produce results (CAMeL-BERT, Baseline v4, potentially Deepseek API), you can:

1. **Compare performance** on same text
2. **Analyze disagreements** — where pipelines disagree
3. **Hybrid approach** — combine strengths of multiple pipelines
4. **Test generalization** — run on other OpenITI texts

---

## Troubleshooting

### Issue: "KeyError: 'probabilities'" in post-processing
**Cause**: Old notebook format used  
**Fix**: Make sure you ran the UPDATED notebook (cells 4, 5, 8)

### Issue: Offset out of bounds
**Cause**: Tokenizer offset mismatch  
**Fix**: Ensure `data/processed/alDarrab_clean.txt` matches what was loaded in Colab

### Issue: Very few boundary tokens (<5%)
**Cause**: Model trained on different domain  
**Fix**: Check that `camelbert_binary_classification_final` is the correct fine-tuned model

---

## Files Reference

| File | Purpose |
|------|---------|
| `notebooks/extract_boundary_tokens_final.ipynb` | Updated Colab notebook |
| `scripts/convert_boundary_tokens_direct.py` | Post-processing (gap=20 clustering) |
| `scripts/baseline_v4.py` | Baseline linguistic segmentation |
| `scripts/compare_pipelines.py` | Multi-pipeline comparison |
| `data/processed/alDarrab_clean.txt` | Input text (cleaned) |
| `results/camelbert_alDarrab_raw_inference.json` | Raw CAMeL-BERT output |
| `results/camelbert_alDarrab_char_boundaries.json` | Processed CAMeL-BERT boundaries |
| `results/baseline_v4_alDarrab_segments.json` | Baseline output |
| `results/comparison_alDarrab.json` | Final comparison metrics |

---

**Status**: Ready for Colab execution  
**Next Step**: Upload alDarrab_clean.txt to Google Drive and run notebook
