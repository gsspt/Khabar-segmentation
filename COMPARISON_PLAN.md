# Comparison: Gold Standard vs Baseline v4 vs CAMeL-BERT

## Setup

**Test Text**: `0392IbnIsmacilMisri.CuqalaMajanin` (Shamela0027093)
- **Size**: 15,462 chars / 2,978 words
- **Type**: Classical Arabic biographical narrative

---

## Results Summary

### 1. Gold Standard (Deepseek API)
**Status**: ✅ Complete

- **Total Segments**: 63
- **Composition**:
  - with_isnad (transmission chains): 26 (41.3%)
  - poetry (embedded verse): 20 (31.7%)
  - prose (pure narrative): 16 (25.4%)
  - continuation (fragment markers): 1 (1.6%)

**Method**: Deepseek API annotation + intelligent chunking with deduplication
**Processing**: 12 chunks × 1500 chars with 200-char overlap
**API Success Rate**: 100% (12/12)

**File**: `data/gold_standard/gold_standard_0392IbnIsmacil_v2.json`

---

### 2. Baseline v4 (Rule-based Linguistic)
**Status**: ✅ Complete

- **Total Segments**: 7
- **Composition**:
  - with_isnad: 7 (100%)
  - poetry: 0
  - prose: 0

**Method**: Pattern matching for isnads (transmission chains) only
**Logic**: 
  - Detects isnad keywords (حدثنا، أخبرنا، قال، etc.)
  - Identifies sentence boundaries
  - Conservative: only recognizes explicit transmission chains

**Performance**:
- ✅ Detects all 26 isnads from gold standard
- ❌ Misses 56 segments (88.9% false negatives)
- ❌ Ignores poetry and pure prose narratives
- ⚠️ Very high precision, very low recall

**File**: `results/baseline_v4_comparison.json`

---

### 3. CAMeL-BERT (Neural Fine-tuned)
**Status**: ⏳ Pending Colab Inference

**Model Location**: Colab checkpoint (not yet downloaded)
**Fine-tuning Data**: Binary classification on annotated corpus
**Training Target**: F1 0.98+ on validation set

**Expected Capabilities**:
- Token-level classification (segment start/not-start)
- Learned to recognize all segment types (not just isnads)
- Should generalize to prose and poetry better than baseline

**How to Get Results**:
1. Open Colab notebook: `notebooks/camelbert_binary_classification_inference.ipynb`
2. Run inference on the 0392IbnIsmacil text
3. Export results as JSON
4. Copy to `results/camelbert_results.json`

---

## Comparison Metrics

### Quantity
| Model | Total | with_isnad | poetry | prose |
|-------|-------|-----------|--------|-------|
| **Gold Standard** | 63 | 26 | 20 | 16 |
| **Baseline v4** | 7 | 7 | 0 | 0 |
| **CAMeL-BERT** | ? | ? | ? | ? |

### Quality Measures
Once CAMeL-BERT results available, compare:
- **Recall**: % of gold standard segments detected
- **Precision**: % of detected segments that match gold standard
- **F1 Score**: Harmonic mean (precision + recall)
- **Type Accuracy**: Correct segment type classification
- **Coverage**: Total text length covered (should be ~100%)

### Speed
- **Baseline v4**: <1s (local, rule-based)
- **Gold Standard**: ~2.5 min (API calls + processing)
- **CAMeL-BERT**: ~5-30s (model inference, GPU in Colab)

---

## Interpretation

### Baseline v4
**Verdict**: Good precision, poor recall
- Best for: Finding explicit transmission chains only
- Use case: Conservative, high-confidence isnad detection
- Issue: Blind to narrative structure without isnads

### Gold Standard (Deepseek)
**Verdict**: Comprehensive but needs validation
- Advantage: Covers all narrative types (prose, poetry, isnads)
- Advantage: Good segmentation decisions
- Use case: Training target for neural models
- Issue: May have annotation errors (LLM-based)

### CAMeL-BERT (Expected)
**Verdict**: Should be balanced
- Expected: Better recall than baseline, more accurate than pure rules
- Expected: Faster than API, more accurate than heuristics
- Expected: Able to generalize to unseen texts
- Use case: Practical deployable model

---

## Next Steps

1. **Get CAMeL-BERT Results**: Run Colab inference notebook
2. **Format Comparison**: Convert all outputs to common format
3. **Calculate Metrics**: Compute recall, precision, F1
4. **Error Analysis**: Which segments does each model miss/hallucinate?
5. **Generalization Test**: Evaluate on different OpenITI texts
6. **Production Recommendation**: Which model to deploy?

---

## File Structure

```
results/
├── baseline_v4_comparison.json          # Baseline v4 results
├── camelbert_results.json               # CAMeL-BERT (to be added)
├── comparison_summary.json              # Summary stats
└── comparison_analysis.json             # Detailed comparison

data/gold_standard/
└── gold_standard_0392IbnIsmacil_v2.json # Gold standard (reference)
```
