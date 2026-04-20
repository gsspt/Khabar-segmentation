# Token-to-Segment Conversion Analysis

## Results from v2 (Boundary Transition Method)

```
Model Output:
  Total tokens: 297,984
  Boundary tokens: 17,938 (6.0%)
  
Boundary Transitions:
  Starts (0→1): 867
  Ends (1→0): 867
  
Extracted Segments:
  Total: 1,441
  Isnads: 742 (51.5%)
  Prose: 699 (48.5%)
  
vs Gold Standard (613 khabars):
  Recall: 235.1%
  Ratio: 2.35x OVER-segments
```

---

## The Core Problem

**Question**: Why does the model identify 867 boundary transitions when there should only be ~300-400?

**Hypothesis**: The model is detecting multiple isnad boundaries **within what gold standard considers a single khabar**.

Example scenario:
```
Gold Standard view (1 khabar):
  "حدثنا محمد قال أخبرنا علي قال حكى الحسن..."
  → 1 segment with multiple transmission chain elements

Model view (multiple boundaries):
  "حدثنا [BOUNDARY] محمد [BOUNDARY] قال [BOUNDARY] 
   أخبرنا [BOUNDARY] علي [BOUNDARY] قال [BOUNDARY]..."
  → Detects 6 boundaries within what's really 1 isnad

Result: 867 detected boundaries vs ~300-400 expected
```

---

## Analysis: What the Model Actually Learned

The 6% boundary ratio (17,938 tokens) suggests the model learned:
- **Token-level isnad markers**: Individual transmission chain components
- **Not khabar-level boundaries**: The higher-level narrative units

The model was trained on segments with internal structure:
```json
{
  "akhbar_id": 1,
  "segments": [
    {"type": "isnad", "text": "حدثنا محمد بن عمر..."},
    {"type": "matn", "text": "قال رأيت..."}
  ]
}
```

But "khabars" in the gold standard might span MULTIPLE of these segments:
```
1 Khabar = 1+ isnads + 1 matn + optional poetry
```

---

## Why V1 (Clustering) Also Over-segmented

V1: Cluster adjacent boundary tokens → 752 clusters → 1,302 segments
V2: Find transitions → 867 transitions → 1,441 segments

Both methods over-segment by similar amounts (2.12x vs 2.35x) because the underlying problem is the same:
- **Model detects 867 isnad events**
- **Gold standard expects 300-400 khabar boundaries**
- **Ratio: ~2.3x difference**

---

## Possible Solutions

### Option 1: Merge Adjacent Isnads
Assumption: Consecutive isnads separated by short prose = single khabar

```python
def merge_adjacent_isnads(segments, min_prose_length=50):
    """
    Merge isnads separated by short prose blocks.
    If prose between isnads < N chars, merge with previous isnad.
    """
    # Current: 867 isnads
    # After merging: ~350-400 isnads (if ~50% of gaps < 50 chars)
```

### Option 2: Use Confidence Threshold
Assumption: High-confidence boundaries are real, low-confidence are false positives

Current: All boundary tokens treated equally
Fixed: Filter tokens by probability before finding transitions

```python
def find_transitions_with_threshold(predictions, probabilities, threshold=0.95):
    """Only count transitions for high-confidence boundary tokens."""
    # Original: 6.0% boundary tokens
    # Filtered at 0.95: ~2-3% boundary tokens
    # Expected transitions: ~300-400
```

### Option 3: Validate Against Baseline v4
Assumption: Baseline v4 is conservative but precise (93.8% recall on 613)

Strategy:
1. Use CAMeL-BERT to find boundaries
2. Only keep boundaries that match or extend Baseline v4 predictions
3. Filter spurious CAMeL-BERT boundaries

```python
def validate_with_baseline(camelbert_boundaries, baseline_boundaries):
    """Keep CAMeL-BERT boundaries that align with Baseline."""
    # Baseline: ~260 isnad boundaries
    # CAMeL-BERT: 867 boundaries
    # After validation: ~350-400 (between both)
```

### Option 4: Return to Segment-Level Classification
Current approach assumes token-level predictions map to segment boundaries.
But the model was trained on **segment-level classification**.

```python
# Original training: "Is this segment an isnad?" (binary per segment)
# Current inference: "Is this token a boundary?" (binary per token)

# These are different tasks!
# Perhaps we should:
# 1. Go back to segment-level approach
# 2. Reconstruct segments from predictions
# 3. Use token probabilities to identify segment boundaries
```

---

## Recommendation: Hybrid Approach

Combine multiple methods:

```python
def smart_segment_extraction(predictions, offsets, probabilities):
    """
    1. Find high-confidence boundary transitions (prob > 0.90)
    2. Merge adjacent isnads separated by short prose (< 50 chars)
    3. Validate against Baseline v4 if available
    4. Result: ~350-400 isnads (closer to 613 khabars)
    """
```

This would:
- Keep high-confidence predictions (reduce false positives)
- Recognize that isnads can span multiple transmission chain elements
- Align with Baseline v4's conservative approach
- Target 55-60% recall instead of 235%

---

## Next Steps

1. **Try Option 2 (confidence threshold)**
   - Implement filtering by `probabilities > 0.90`
   - Measure effect on segment count
   - Quick test to see if high-confidence boundaries are fewer but better

2. **Try Option 1 (merge adjacent isnads)**
   - Merge isnads with < 50 char prose between them
   - Test merge thresholds (25, 50, 100 chars)
   - See if merging brings count closer to 613

3. **Hybrid: Confidence + Merging**
   - Filter high-confidence boundaries
   - Merge adjacent isnads
   - Target: 55-65% recall (340-400 segments)

---

## Summary

The boundary transition method is **working as designed**, but the model's definition of "boundary" is **sub-khabar level**.

The 2.35x over-segmentation is not a bug in the conversion logic—it's a **mismatch between what the model learned and what we're trying to measure**.

Next phase: Refine the extraction logic to handle this mismatch through confidence filtering, merging, or hybrid validation approaches.
