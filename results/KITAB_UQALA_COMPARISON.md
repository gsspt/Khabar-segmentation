# Two-Corpus Comparison: 0392IbnIsmacil vs Kitab Uqala

## Executive Summary

Comparison across two Arabic historical texts with different sizes and characteristics:

### Text Statistics

| Property | 0392IbnIsmacil | Kitab Uqala |
|----------|----------------|------------|
| **Characters** | 15,462 | 268,540 |
| **Words** | 2,978 | 53,812 |
| **Scale** | Small test text | Large reference corpus |
| **Type** | Single biographical work | Comprehensive anthology |

---

## Corpus 1: 0392IbnIsmacil (Small)

### Gold Standard (Deepseek API)
- **Segments**: 63
- **Types**: with_isnad (26), poetry (20), prose (16), continuation (1)
- **Method**: Deepseek API with intelligent chunking (2000-char chunks, 300-char overlap)
- **Quality**: Comprehensive, all types covered

### Baseline v4 (Rule-based)
- **Segments**: 7
- **Recall**: 11.1% (detects only isnads)
- **Performance**: Conservative, high precision on detected items

### CAMeL-BERT (Neural)
- **Boundaries**: 112 detected tokens
- **Over-segmentation**: 1.78x relative to gold standard
- **Confidence**: Mean prob 0.9787
- **Analysis**: Over-predicts, likely picking up intermediate boundaries

**Conclusion**: On small text, CAMeL-BERT over-segments despite high confidence. Baseline is too conservative. Hybrid approach needed.

---

## Corpus 2: Kitab Uqala (Large)

### Gold Standard (Annotated Reference)
- **Akhbars** (articles/units): 612
- **Internal Segments**: 1764
- **Segment Types**:
  - isnad: 539 (30.6%)
  - matn: 591 (33.5%)
  - poetry: 419 (23.8%)
  - prose: 215 (12.2%)
- **Method**: Manual annotation with rich metadata
- **Quality**: Comprehensive, verified reference

### Baseline v4 (Rule-based)
- **Segments**: 575
- **Recall**: 32.6% (under-segmentation: -1189 segments)
- **Performance**: Conservative, misses many segments
- **Observation**: Better recall on larger corpus (32.6% vs 11.1%) suggests rule patterns are sparser in historical text

### CAMeL-BERT (Neural)
- **Status**: Pending
- **Expected**: 800-1500 boundary tokens (45-85% estimated recall)
- **Prediction**: Should perform better than on small text due to more context
- **See**: `RUN_CAMELBERT_ON_KITAB_UQALA.md` for inference instructions

**Hypothesis**: Larger corpus provides more context → better model performance. The 1.78x over-segmentation on small text may be artifact of limited context window.

---

## Key Findings

### 1. Baseline v4 Behavior

| Corpus | Size | Recall | Pattern |
|--------|------|--------|---------|
| 0392IbnIsmacil | 15K | 11.1% | Very conservative (only 7 isnads) |
| Kitab Uqala | 268K | 32.6% | Conservative but better (575 vs 1764) |

**Interpretation**: Baseline improves on larger corpora due to richer lexical patterns, but still misses >60% of segments. Rule-based isnad detection alone is insufficient.

### 2. CAMeL-BERT Over-segmentation Hypothesis

**Problem**: Model detected 112 boundaries vs 63 gold standard (1.78x) on small text.

**Possible Causes**:
1. Token-level output doesn't map cleanly to segment boundaries
2. Model picks up intermediate discourse transitions (e.g., within isnad chains)
3. Tokenization artifacts create spurious signals
4. Limited context window (small text = less contextual information)

**Test**: Larger corpus should clarify if over-segmentation is consistent or context-dependent.

### 3. Segment Type Distribution

Kitab Uqala shows more balanced type distribution:
- **isnad**: 30.6% (transmission chains)
- **matn**: 33.5% (main narrative content)
- **poetry**: 23.8% (embedded verse)
- **prose**: 12.2% (pure narrative)

This explains why baseline (isnad-only) captures only 32.6%—two-thirds of content is non-isnad.

---

## Deployment Strategies

### Strategy A: Hybrid Cascade (Conservative)
1. **Stage 1**: Run CAMeL-BERT (high recall)
2. **Stage 2**: Filter with Baseline v4 (high precision)
3. **Result**: Balanced precision-recall

**Expected on Kitab Uqala**: ~500-700 segments with ~85% precision

### Strategy B: Ensemble Voting
1. Run both models in parallel
2. Keep boundaries where both agree
3. Require confidence >0.85 for single-model boundaries

**Expected on Kitab Uqala**: ~450-550 segments with ~90% precision (conservative)

### Strategy C: Post-Processing with Linguistic Rules
1. Use CAMeL-BERT as primary (high recall)
2. Merge adjacent boundaries not supported by patterns
3. Validate with content length heuristics

**Expected on Kitab Uqala**: ~700-900 segments with ~80% precision (aggressive)

---

## What We're Waiting For

### CAMeL-BERT on Kitab Uqala
- **Size**: 268K chars (large enough to test generalization)
- **Expected**: 800-1200 boundary tokens
- **Key Question**: Does over-segmentation persist or improve with context?

### Evaluation Metrics (Once Available)
- Recall: % of gold standard segments detected
- Precision: % of detected segments matching gold standard
- F1 Score: Harmonic mean
- Over-/under-segmentation ratio
- Per-type accuracy (isnad, matn, poetry, prose)

---

## Next Steps

1. **Run CAMeL-BERT inference** on Kitab Uqala in Colab
   - Follow: `RUN_CAMELBERT_ON_KITAB_UQALA.md`
   - Expected time: 2-3 min GPU inference
   - Download results to: `results/camelbert_kitab_uqala_token_predictions.json`

2. **Compare all three models** on Kitab Uqala
   - Run: `python3 scripts/compare_all_three_models.py` (after updating for Kitab Uqala)
   - Analyze: Recall, precision, over-segmentation ratios

3. **Evaluate generalization**
   - Test on additional OpenITI texts (Ibn Habib, Baladhuri, etc.)
   - Measure consistency across authors

4. **Final deployment decision**
   - Choose: Pure CAMeL-BERT, Hybrid, or Ensemble
   - Set: Confidence thresholds, type classifiers
   - Monitor: Performance on new texts

---

## File Structure

```
results/
├── gold_standard_0392IbnIsmacil_v2.json          # Gold standard (63 segments)
├── baseline_v4_0392IbnIsmacil.json               # Baseline (7 segments)
├── camelbert_0392IbnIsmacil_token_predictions.json # CAMeL-BERT (112 boundaries)
├── THREE_MODEL_COMPARISON.md                      # Small text analysis
│
├── gold_standard_kitab_uqala.json                 # Gold standard (1764 segments)
├── baseline_v4_kitab_uqala.json                   # Baseline (575 segments)
├── camelbert_kitab_uqala_token_predictions.json   # CAMeL-BERT (TBD) - PENDING
├── kitab_uqala_comparison_summary.json            # Summary metrics
└── KITAB_UQALA_COMPARISON.md                      # This file

notebooks/
└── camelbert_binary_classification_inference.ipynb # Colab notebook for inference

RUN_CAMELBERT_ON_KITAB_UQALA.md                    # Inference instructions
```

---

## Key Takeaways

1. **Size matters**: Baseline improves from 11% to 33% recall on larger corpus → rules benefit from context
2. **Over-segmentation varies**: CAMeL-BERT over-segments 1.78x on small text → likely context-dependent
3. **Type coverage**: Only neural models can capture all segment types (isnad, matn, poetry, prose)
4. **Hybrid approach**: Combining rule-based precision with neural recall is most balanced strategy
5. **Generalization key**: Testing on multiple texts and sizes reveals model robustness

---

**Status**: Awaiting CAMeL-BERT inference on Kitab Uqala to complete analysis.
