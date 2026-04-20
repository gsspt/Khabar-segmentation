# CAMeL-BERT Inference on Kitab Uqala (Large Corpus)

## Overview

This guide runs CAMeL-BERT on the much larger Kitab Uqala corpus (268K chars, 53K words, 612 akhbars) to evaluate generalization beyond the smaller 0392IbnIsmacil test text.

**Expected Results:**
- Gold Standard: 1764 internal segments
- Baseline v4: 575 segments (32.6% recall)
- CAMeL-BERT: TBD (expected 800-1500 boundary tokens)

---

## Step 1: Prepare Input Text

The corpus is already prepared at:
```
data/processed/kitab_uqala_reference_corpus.txt
```

Size: 268,540 chars / 53,812 words

---

## Step 2: Run Inference in Colab

### Option A: Use Existing CAMeL-BERT Checkpoint (Recommended)

1. Open: `notebooks/camelbert_binary_classification_inference.ipynb` in Colab
2. Run cells 1-7 to load the model
3. Add this inference cell **after cell 7**:

```python
# ============================================================================
# INFERENCE ON KITAB UQALA (LARGE CORPUS)
# ============================================================================

from pathlib import Path
import re
import json

# Load the large corpus
corpus_path = Path('data/processed/kitab_uqala_reference_corpus.txt')
raw_text = corpus_path.read_text(encoding='utf-8')

print(f"[INFO] Text loaded: {len(raw_text)} chars, {len(raw_text.split())} words")

# Clean whitespace
raw_text = re.sub(r'[ \t]+', ' ', raw_text)
raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
cleaned_text = raw_text.strip()

print(f"[INFO] After cleaning: {len(cleaned_text)} chars")

# Run inference
print(f"[INFO] Running CAMeL-BERT inference on Kitab Uqala...")

# For large texts, use chunking
results = predict_boundaries(cleaned_text, tokenizer, model, threshold=0.5, chunk_size=512)

print(f"\n[RESULTS] Token-level predictions:")
print(f"  Total tokens: {len(results['tokens'])}")
print(f"  Boundary tokens: {len(results['boundary_indices'])}")
print(f"  Boundary ratio: {len(results['boundary_indices']) / len(results['tokens']) * 100:.1f}%")

# Save results
import json
output = {
    'model': 'camelbert_binary_classification_final',
    'text_info': {
        'file': 'kitab_uqala_reference_corpus',
        'chars': len(cleaned_text),
        'words': len(cleaned_text.split())
    },
    'token_predictions': {
        'total_tokens': len(results['tokens']),
        'boundary_tokens': len(results['boundary_indices']),
        'boundary_indices': results['boundary_indices']
    },
    'probabilities': {
        'mean_prob_boundary': float(np.mean([results['probabilities'][i] for i in results['boundary_indices'][:500]])) if results['boundary_indices'] else 0,
        'mean_prob_non_boundary': float(np.mean([results['probabilities'][i] for i in range(1, min(500, len(results['tokens']))) if i not in results['boundary_indices'][:100]])) if len(results['tokens']) > 1 else 0
    }
}

# Save to Drive
output_file = Path('results/camelbert_kitab_uqala_token_predictions.json')
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Results saved to {output_file}")
```

---

## Step 3: Download Results

1. In Colab Files panel (left sidebar)
2. Navigate to `results/camelbert_kitab_uqala_token_predictions.json`
3. Right-click → Download
4. Save to local `results/` directory

---

## Step 4: Run Local Comparison

After downloading the file, run:

```bash
python3 << 'EOF'
import json

# Load all three models
with open('results/gold_standard_kitab_uqala.json', 'r', encoding='utf-8') as f:
    gold_std = json.load(f)

with open('results/baseline_v4_kitab_uqala.json', 'r', encoding='utf-8') as f:
    baseline = json.load(f)

with open('results/camelbert_kitab_uqala_token_predictions.json', 'r', encoding='utf-8') as f:
    camelbert = json.load(f)

# Print comparison
print("\n" + "="*80)
print("KITAB UQALA - THREE-MODEL COMPARISON")
print("="*80)

print(f"\n{'Model':<20} | {'Segments':<15} | {'Key Metrics':<30}")
print("-"*80)

gold_total = gold_std['total_segments']
baseline_total = baseline['total_segments']
camelbert_boundaries = len(camelbert['token_predictions']['boundary_indices'])

print(f"{'Gold Standard':<20} | {gold_total:<15} | Annotated reference (1764 segs)")
print(f"{'Baseline v4':<20} | {baseline_total:<15} | Recall: {baseline_total/gold_total*100:.1f}%")
print(f"{'CAMeL-BERT':<20} | {camelbert_boundaries:<15} | Confidence: {camelbert['probabilities']['mean_prob_boundary']:.4f}")

# Recall metrics
print(f"\n[METRICS]")
print(f"  Baseline recall: {baseline_total/gold_total*100:.1f}%")
print(f"  CAMeL-BERT estimated recall: {camelbert_boundaries/gold_total*100:.1f}%")
print(f"  Over-segmentation ratio (CAMeL-BERT): {camelbert_boundaries/gold_total:.2f}x")

EOF
```

---

## Expected Output

Assuming CAMeL-BERT detects 1000-1200 boundary tokens:

```
Gold Standard:  1764 segments (reference)
Baseline v4:    575 segments (32.6% recall)
CAMeL-BERT:     ~1100 boundaries (62% estimated recall, 0.62x)
```

This would show CAMeL-BERT performing **much better** on the larger corpus than on the small test (where it over-segmented 1.78x). The larger corpus provides better context for the model.

---

## Notes

- **Large corpus advantage**: More context helps the model distinguish real vs spurious boundaries
- **Training match**: Model was trained on similar large historical text, so generalization should be good
- **Processing time**: 268K chars = ~2-3 min GPU inference (Colab)
- **Memory**: Should be fine on free Colab GPUs

---

## Troubleshooting

**Error: "out of memory"**
- Reduce chunk_size to 256 in `predict_boundaries()`

**Error: "file not found"**
- Ensure `data/processed/kitab_uqala_reference_corpus.txt` is in your Colab workspace
- Mount Google Drive if files are there

**Slow inference**
- Expected: ~2-3 min for 268K chars
- Large corpus takes longer, this is normal
