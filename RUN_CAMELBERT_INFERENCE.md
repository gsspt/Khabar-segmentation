# CAMeL-BERT Inference on 0392IbnIsmacil Text

## Quick Start - Run in Colab

### Step 1: Open the Inference Notebook
1. Go to: `notebooks/camelbert_binary_classification_inference.ipynb`
2. Open it in Google Colab: Click "Open in Colab" button

### Step 2: Run Setup Cells
Execute cells 1-2 to:
- Mount Google Drive
- Install dependencies
- Load the fine-tuned model

### Step 3: Run Inference on Our Test Text

Add this cell **after cell 7** (after model is loaded):

```python
# ============================================================================
# INFERENCE ON 0392IbnIsmacil TEXT
# ============================================================================

# Load the test text
from pathlib import Path
import re

# Read the OpenITI file
openiti_file = Path('openiti_corpus/data/0392IbnIsmacilMisri/0392IbnIsmacilMisri.CuqalaMajanin/0392IbnIsmacilMisri.CuqalaMajanin.Shamela0027093-ara1')
raw_text = openiti_file.read_text(encoding='utf-8')

# Clean it (same as in gold standard generation)
if '#META#Header#End#' in raw_text:
    raw_text = raw_text[raw_text.index('#META#Header#End#') + len('#META#Header#End#'):]

raw_text = re.sub(r'^###[^\n]*\n', '', raw_text, flags=re.MULTILINE)
raw_text = re.sub(r'^#+\s*', '', raw_text, flags=re.MULTILINE)
raw_text = re.sub(r'\n~~', ' ', raw_text)
raw_text = re.sub(r'PageV\d+P\d+', '', raw_text)
raw_text = re.sub(r'\bms\d+\b', '', raw_text)
raw_text = re.sub(r'\[\d+[a-z]?\]', '', raw_text)
raw_text = re.sub(r'[ \t]+', ' ', raw_text)
raw_text = re.sub(r'\n{3,}', '\n\n', raw_text)
cleaned_text = raw_text.strip()

print(f"[INFO] Text loaded: {len(cleaned_text)} chars, {len(cleaned_text.split())} words")

# Run inference
print(f"[INFO] Running CAMeL-BERT inference...")

# The text is long, so split into chunks and predict boundaries
results = predict_boundaries(cleaned_text, tokenizer, model, threshold=0.5)

print(f"\n[RESULTS] Token-level predictions:")
print(f"  Total tokens: {len(results['tokens'])}")
print(f"  Boundary tokens: {len(results['boundary_indices'])}")
print(f"  Boundary ratio: {len(results['boundary_indices']) / len(results['tokens']) * 100:.1f}%")

# Save raw predictions
import json
output = {
    'model': 'camelbert_binary_classification_final',
    'text_info': {
        'file': '0392IbnIsmacilMisri.CuqalaMajanin',
        'chars': len(cleaned_text),
        'words': len(cleaned_text.split())
    },
    'token_predictions': {
        'total_tokens': len(results['tokens']),
        'boundary_tokens': len(results['boundary_indices']),
        'boundary_indices': results['boundary_indices'][:200],  # Save first 200
    },
    'probabilities': {
        'mean_prob_boundary': float(np.mean([results['probabilities'][i] for i in results['boundary_indices'][:100]])),
        'mean_prob_non_boundary': float(np.mean([results['probabilities'][i] for i in range(1, min(100, len(results['tokens']))) if i not in results['boundary_indices']]))
    }
}

# Save to Drive
output_file = Path('results/camelbert_0392IbnIsmacil_token_predictions.json')
output_file.parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Results saved to {output_file}")
```

### Step 4: Download Results

1. In Colab, go to Files panel (left sidebar)
2. Navigate to `results/camelbert_0392IbnIsmacil_token_predictions.json`
3. Right-click → Download
4. Save it to your local `results/` directory

### Step 5: Run Local Comparison

After downloading the file, run locally:

```bash
python3 compare_all_three.py
```

---

## Alternative: Download Model and Run Locally

If you want to run inference locally without Colab:

1. Download the model from Colab:
   ```bash
   # In Colab terminal
   !zip -r /tmp/camelbert_model.zip checkpoints/camelbert_binary_classification_final/
   # Download the zip from Files panel
   ```

2. Extract locally:
   ```bash
   unzip camelbert_model.zip -d .
   ```

3. Run local inference:
   ```bash
   python3 scripts/inference/run_camelbert_local.py
   ```

---

## Expected Output

The CAMeL-BERT model should predict **60-80 boundary tokens** (segment starts) in the 0392IbnIsmacil text.

**Comparison with Gold Standard (63 segments) and Baseline v4 (7 segments)** will show:
- ✅ CAMeL-BERT: Better than baseline, close to gold standard
- ❌ Baseline: Only detects isnads (7 segments)
- 🏆 Gold Standard: Comprehensive reference (63 segments)
