# Split Colab/Local Workflow for CAMeL-BERT Processing

## Overview

Split the work where it makes sense:

| Task | Location | Reason |
|------|----------|--------|
| **Inference** | Colab (GPU) | 268K chars needs GPU, takes 5-10 min |
| **Post-processing** | Local (CPU) | Lightweight, instant results |
| **Analysis** | Local (CPU) | Compare with baselines, generate metrics |

---

## Workflow Steps

### Step 1: Run Inference in Colab (5-10 minutes)

**Open Notebook**: `notebooks/camelbert_inference_for_export.ipynb`

1. Mount Drive
2. Load model (CAMeL-BERT checkpoint)
3. Load Kitab Uqala text
4. Run inference with offset mapping (GPU accelerated)
5. Save results to `results/camelbert_kitab_uqala_raw_inference.json`

**Output**: JSON file with:
- `predictions`: Token-level binary predictions (0/1)
- `probabilities`: Confidence scores
- `tokens`: Token strings (for debugging)
- `offsets`: **Character positions for each token** ← Critical!

**File size**: ~10-20 MB

---

### Step 2: Download Results from Colab

1. In Colab Files panel (left sidebar) → click **Refresh**
2. Navigate to: `results/camelbert_kitab_uqala_raw_inference.json`
3. Right-click → **Download**
4. Save to your local: `results/camelbert_kitab_uqala_raw_inference.json`

---

### Step 3: Local Post-Processing (30 seconds)

Run locally on your machine:

```bash
python3 scripts/camelbert_local_postprocess.py \
    --input results/camelbert_kitab_uqala_raw_inference.json \
    --output results/camelbert_kitab_uqala_segments.json \
    --text data/processed/kitab_uqala_reference_corpus.txt \
    --gold-standard 613
```

**What it does:**
1. **Cluster adjacent boundary tokens** → actual segment boundaries
2. **Extract text segments** → cut document at boundaries
3. **Compare with gold standard** → calculate recall
4. **Save results** → JSON with segments + metrics

**Output**: `results/camelbert_kitab_uqala_segments.json`

---

### Step 4: Analyze Results Locally

Check the output JSON:

```bash
# View summary
cat results/camelbert_kitab_uqala_segments.json | head -100

# Python analysis
python3 << 'EOF'
import json

with open('results/camelbert_kitab_uqala_segments.json') as f:
    data = json.load(f)

print(f"Total segments: {data['segmentation']['total_segments']}")
print(f"Recall vs gold: {data['evaluation']['recall']}")
print(f"Type breakdown: {data['segmentation']['type_breakdown']}")
EOF
```

---

## What Each File Contains

### Input: Raw Inference JSON (from Colab)

```json
{
  "metadata": {
    "corpus": "kitab_uqala_reference_corpus",
    "text_size_chars": 268540,
    "model": "camelbert_binary_classification_final"
  },
  "inference_results": {
    "total_tokens": 512,
    "boundary_tokens": 200,
    "predictions": [0, 0, 1, 1, 0, ...],  // Token-level binary predictions
    "probabilities": [0.05, 0.02, 0.98, 0.97, ...],  // Confidence scores
    "tokens": ["[CLS]", "حدثنا", "محمد", ...],  // Token strings
    "offsets": [[0,0], [0,5], [6,11], ...]  // Char positions: [start, end]
  }
}
```

### Output: Segmented Results (from Local)

```json
{
  "metadata": {
    "source": "results/camelbert_kitab_uqala_raw_inference.json",
    "text": "data/processed/kitab_uqala_reference_corpus.txt",
    "gold_standard": 613
  },
  "segmentation": {
    "segments": [
      {
        "text": "حدثنا محمد بن عمر",
        "start": 0,
        "end": 18,
        "type": "isnad",
        "length": 18
      },
      {
        "text": "قال رأيت النبي...",
        "start": 18,
        "end": 150,
        "type": "prose",
        "length": 132
      }
    ],
    "total_segments": 450,
    "type_breakdown": {
      "isnad": 200,
      "prose": 250
    }
  },
  "evaluation": {
    "gold_standard": 613,
    "camelbert_segments": 450,
    "recall": "73.4%",
    "ratio": "0.73x",
    "assessment": "UNDER-segments"
  }
}
```

---

## Workflow Diagram

```
Colab GPU                          Local CPU
─────────────────────────────────────────────────────

[Model Checkpoint]
         ↓
[Load Kitab Uqala (268K chars)]
         ↓
[Run Inference with GPU] ← Fast (5-10 min)
         ↓
[Save JSON with offsets]
         ↓
    ← Download File →
         ↓
      [Local Post-Processing] ← Fast (30 sec)
         ↓
    [Extract Segments]
         ↓
    [Compare with 613 Khabars]
         ↓
    [Generate Metrics]
         ↓
    [Save Results JSON]
         ↓
    [Analyze & Compare with Baseline v4]
```

---

## Expected Performance

| Stage | Metric | Expected |
|-------|--------|----------|
| Token Classification | Accuracy | 99.93% (from Colab) |
| Token Clustering | Boundary clusters | ~200-250 |
| Segment Extraction | Total segments | 400-500 (vs 613 gold) |
| Final Recall | vs Gold Standard | 65-75% |

---

## Time Breakdown

| Task | Location | Time |
|------|----------|------|
| Colab inference | GPU | 5-10 min |
| Download | Network | 1-2 min |
| Local post-processing | CPU | 30 sec |
| **Total** | - | **7-13 min** |

---

## Commands Reference

### Colab Notebook

```python
# Run all cells in: notebooks/camelbert_inference_for_export.ipynb
```

### Local Script

```bash
# Basic usage (uses defaults)
python3 scripts/camelbert_local_postprocess.py \
    --input results/camelbert_kitab_uqala_raw_inference.json \
    --output results/camelbert_kitab_uqala_segments.json

# Full options
python3 scripts/camelbert_local_postprocess.py \
    --input results/camelbert_kitab_uqala_raw_inference.json \
    --output results/camelbert_kitab_uqala_segments.json \
    --text data/processed/kitab_uqala_reference_corpus.txt \
    --gold-standard 613
```

---

## Files Involved

### Colab
- `notebooks/camelbert_inference_for_export.ipynb` ← Run this
- `checkpoints/camelbert_binary_classification_final/` (already on Drive)
- `data/processed/kitab_uqala_reference_corpus.txt` (already on Drive)
- Output: `results/camelbert_kitab_uqala_raw_inference.json` (download this)

### Local
- `scripts/camelbert_local_postprocess.py` ← Run this
- `data/processed/kitab_uqala_reference_corpus.txt` (already local)
- Input: `results/camelbert_kitab_uqala_raw_inference.json` (download from Colab)
- Output: `results/camelbert_kitab_uqala_segments.json` (generated)

---

## Next: Compare with Baseline

Once you have `camelbert_kitab_uqala_segments.json`, compare with Baseline v4:

```bash
python3 << 'EOF'
import json

# Load CAMeL-BERT results
with open('results/camelbert_kitab_uqala_segments.json') as f:
    camelbert = json.load(f)

# Load baseline results
with open('results/baseline_v4_kitab_uqala.json') as f:
    baseline = json.load(f)

gold = 613

print("="*80)
print("COMPARISON")
print("="*80)
print(f"\nGold Standard:    {gold}")
print(f"Baseline v4:      {baseline['total_segments']:3d} ({baseline['total_segments']/gold*100:.1f}% recall)")
print(f"CAMeL-BERT:       {camelbert['segmentation']['total_segments']:3d} ({camelbert['segmentation']['total_segments']/gold*100:.1f}% recall)")
print(f"\n{camelbert['evaluation']['assessment']}")
EOF
```

---

## Troubleshooting

**Problem**: "Input file not found"
- Make sure you downloaded the JSON from Colab to `results/`

**Problem**: "Text file not found"
- Ensure `data/processed/kitab_uqala_reference_corpus.txt` exists locally

**Problem**: Colab inference is slow
- Normal for 268K chars on free Colab GPU (5-10 min)
- Pro GPUs are faster

**Problem**: Post-processing gives wrong segment count
- Check token offsets are valid (should be character positions 0-268540)
- Verify text file matches what was used in Colab

---

## Advanced: Process Multiple Texts

Once you have the setup, you can run it on any text:

```bash
# Process different corpus
python3 scripts/camelbert_local_postprocess.py \
    --input results/camelbert_ibn_habib_raw_inference.json \
    --output results/camelbert_ibn_habib_segments.json \
    --text data/processed/ibn_habib_corpus.txt \
    --gold-standard 708  # Different gold standard count
```

Just re-run the Colab notebook with different input text.
