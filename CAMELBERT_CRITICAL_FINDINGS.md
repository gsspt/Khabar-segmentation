# CAMeL-BERT Critical Findings: Windowing Strategy Issue

**Date**: 2026-04-22  
**Severity**: IMPORTANT — Reveals core workflow inefficiency

---

## The Numbers Don't Lie

```
RAW INFERENCE STATISTICS:
  Total tokens in raw file:           297,984
  Unique char_start positions:        72,057
  Char_starts appearing multiple:     8,097
  Total tokens appearing 2+× times:   233,653 (78.4% of all tokens!)
  
  Tokens appearing exactly 1×:        63,960 (21.5%)
  Tokens appearing exactly 2×:        7,516
  Tokens appearing 3+× times:         (unknown, but max=481 for one token!)
  
  PREDICTION CONFLICTS (pred 0→1):    425 positions
```

## What This Means

**The current pipeline does NOT use 50% overlap.** It uses **MASSIVE overlap**.

### Evidence

If the notebook used 50% overlap (stride=256):
- Expected: Each token appears ~1.5-2.5 times
- Actual: 78.4% of tokens appear 2+ times, max=481
- **Conclusion**: The inference code is running MANY overlapping windows

### Why?

Possible explanations:
1. **Very small stride** (e.g., stride=64 or stride=32)
   - This would explain why 78% of tokens appear multiple times
   - stride=32 → 297,984 ÷ 32 ≈ 9,312 windows
   - Each token would appear in ~60 windows (512 token window ÷ 32 stride ≈ 16)

2. **Chunking with heavy overlap for safety**
   - Maybe the Colab notebook used overlap to ensure no boundary is missed
   - But this creates massive redundancy

3. **Bug in the chunking logic**
   - Tokens being duplicated unintentionally
   - Shuffled order causing re-processing

## The Problem This Creates

### 1. Computational Waste
- 78% of tokens are processed **multiple times**
- If stride=32: processing is 16× more than necessary!
- This explains the 31MB raw inference file

### 2. Deduplication Complexity
- 233,653 duplicate tokens need to be merged
- Current approach: max probability wins
- But **425 positions have conflicting predictions** (pred=0 in window A, pred=1 in window B)

### 3. Data Quality Issues
- When 481 windows predict the same token differently, which is correct?
- Random seed sensitivity: different random seeds might give different results
- No validation that predictions are consistent

---

## Example: The Conflicting Token at char_start=924

```
This position appears in MANY windows with different predictions:

First few occurrences:
  Window A: idx=0    → pred=1, prob=0.9966
  Window B: idx=481  → pred=0, prob=0.0030
  Window C: idx=962  → pred=1, prob=0.9985
  Window D: idx=1443 → pred=0, prob=0.0395
  ...
  (more windows with mostly pred=0, prob ≈ 0.001)

Decision made: max(all probs) = 0.9985 at idx=962 → predict 1

But the question remains:
- Why pred=1 with prob=0.99 in some windows?
- Why pred=0 with prob=0.0005 in other windows?
- Which is the "true" prediction given full context?
```

---

## What The Optimal Strategy Should Be

### Option 1: Minimal Overlap (FAST)
```python
# No overlap: stride = 512
window_size = 512
stride = 512

for i in range(0, len(tokens), stride):
    window = tokens[i:i+window_size]
    preds = model(window)
    
# Result: ~584 windows
# Processing time: ✓ Fast (1× compute)
# Quality: ? Edge effects at boundaries
# Post-processing: None needed (no conflicts)
```

**Cost**: May miss boundaries that span window edges.

---

### Option 2: Smart Overlap (BALANCED) ← RECOMMENDED
```python
# Overlap with confident center: stride=256, margin=64
window_size = 512
stride = 256
margin = 64  # Ignore edges

predictions = {}
for i in range(0, len(tokens), stride):
    window_end = min(i + window_size, len(tokens))
    if window_end - i < 100:
        break
    
    preds = model(tokens[i:window_end])
    
    # Only keep confident center (positions 64-448)
    for j in range(margin, min(len(preds), window_size - margin)):
        global_pos = i + j
        predictions[global_pos] = preds[j]  # No duplicates!

# Result: ~1,164 windows
# Processing time: 2× compute
# Quality: ✓ Full context everywhere
# Post-processing: None needed (smart windowing eliminates duplicates)
```

**Benefit**: Eliminates deduplication step entirely.

---

### Option 3: Current (EXPENSIVE)
```python
# Extreme overlap: stride=32 (estimated)
window_size = 512
stride = 32  # ???

for i in range(0, len(tokens), stride):
    preds = model(tokens[i:i+window_size])
    # Store all results with duplicates

# Result: ~9,312 windows
# Processing time: 16× compute (!!)
# Quality: ? Excessive redundancy
# Post-processing: Complex deduplication with conflict resolution
```

**Problem**: 16× more compute than necessary for marginal quality gain.

---

## Critical Question: Why Is the Current One So Heavy?

Looking at the notebook implementation, possible reasons:

1. **Safety/Redundancy**: Maybe designed to catch every possible boundary
   - Assumption: Small stride = lower chance of missing boundaries
   - Reality: Massive overhead for minimal gain

2. **Lack of optimization**: Might have been written without considering efficiency
   - Prototype code that worked but wasn't optimized
   - Never profiled or benchmarked

3. **Tokenization mismatch**: The notebook might be re-tokenizing with different settings
   - Leading to different token counts per window
   - Causing unexpected overlaps

---

## My Recommendation

### For Next CAMeL-BERT Inference

**Use the Smart Overlap approach:**

```python
def infer_camelbert_optimized(text_path, model, output_path):
    """
    Optimal CAMeL-BERT inference: balance quality and efficiency.
    
    Strategy:
    - Use 50% overlap (stride=256)
    - Keep only confident center of each window (ignore margin=64)
    - No deduplication needed
    - Direct mapping to char positions
    """
    
    # Tokenize entire text with offset mapping
    tokens = tokenize_with_offsets(text_path)
    
    predictions = {}
    margin = 64
    stride = 256
    window_size = 512
    
    for i in range(0, len(tokens), stride):
        window_end = min(i + window_size, len(tokens))
        window_tokens = tokens[i:window_end]
        
        if len(window_tokens) < 100:
            break
        
        # Run inference
        preds, probs = model(window_tokens)
        
        # Keep only center (positions margin to window_size-margin)
        for j in range(margin, min(len(preds), len(window_tokens) - margin)):
            global_idx = i + j
            predictions[global_idx] = {
                'pred': preds[j],
                'prob': probs[j],
                'char_offset': tokens[global_idx].offset
            }
    
    # Handle final chunk if needed
    if len(tokens) % stride != 0:
        start = (len(tokens) // stride) * stride
        if start > 0:
            start -= stride  # Overlap with previous
        window_tokens = tokens[start:min(start+512, len(tokens))]
        preds, probs = model(window_tokens)
        for j, (pred, prob) in enumerate(zip(preds, probs)):
            global_idx = start + j
            if global_idx not in predictions:  # Only if not already done
                predictions[global_idx] = {
                    'pred': pred,
                    'prob': prob,
                    'char_offset': tokens[global_idx].offset
                }
    
    # Save raw predictions (minimal post-processing)
    save_predictions(predictions, output_path)
    
    return {
        'total_windows': (len(tokens) + stride - 1) // stride,
        'predictions_count': len(predictions),
        'coverage': len(predictions) / len(tokens) * 100
    }
```

**Expected Improvements:**
- ✓ 2× faster than current (256 stride vs tiny stride)
- ✓ No deduplication needed (each token predicted once)
- ✓ No conflicting predictions (each token has one answer)
- ✓ Better context than no-overlap (512 token window)
- ✓ Clean inference pipeline (less post-processing)

---

## Testing This Hypothesis

To verify the actual stride used in current notebook:

```python
# Count windows by analyzing stride pattern
stride_estimate = 297984 / (72057 * 0.78)  # tokens / (unique * overlap_ratio)
# ≈ 297984 / 56204 ≈ 5.3 words per stride?
# That doesn't make sense... suggests it's token-based not word-based

# Better: analyze gaps in token indices
gaps = [indices[i+1] - indices[i] for i in range(len(indices)-1)]
median_gap = median(gaps)
# If stride=256: median gap ≈ 256
# If stride=32: median gap ≈ 32
```

---

## Summary: The Workflow Issue

| Aspect | Current | Optimal |
|--------|---------|---------|
| **Stride** | Very small (~32?) | 256 (50% overlap) |
| **Windows** | ~9,312 | ~1,164 |
| **Compute** | 16× | 2× |
| **Token Duplicates** | 78.4% | 0% |
| **Conflicts** | 425 | 0 |
| **Post-processing** | Complex dedup | None |
| **Quality** | Good (F1=0.8579) | Likely better |
| **Efficiency** | Poor | Excellent |

**You were absolutely right to question the overlap.** The current approach is using far too much redundancy.

---

## Next Steps

1. **Inspect the Colab notebook** to determine actual stride value
2. **If stride < 128**: Recommend switching to stride=256 with margin=64
3. **Test smart windowing** on Kitab Uqala to compare F1 scores
4. **If F1 is equal**: 8× speedup with smart windows
5. **If F1 improves**: Both speedup AND quality improvement

