# CAMeL-BERT Window Strategy Analysis: Overlap vs. No-Overlap

**Date**: 2026-04-22  
**Context**: Analyze why overlap is needed and what's the optimal inference strategy

---

## 1. THE CORE QUESTION

You're right: CAMeL-BERT can process text in 512-token windows and give boundary predictions for each token. **So why do we need overlap?**

Let's break this down step-by-step.

---

## 2. WHAT BERT TOKEN CLASSIFICATION DOES

### Architecture
```
Input: Sequence of tokens (≤512)
    ↓
[CLS] token1 token2 ... tokenN [SEP]
    ↓
BERT Encoder (12 layers of self-attention)
    ↓
Output: Logits for each token (binary classification: boundary/non-boundary)
    ↓
Predictions: [0, 1, 0, 1, 0, ...]
```

### Key Point: Context Dependency
- Each token's prediction depends on **ALL other tokens in the same window** (via self-attention)
- Token at position 512 has context from tokens 1-511
- Token at position 1 has context from tokens 2-512

---

## 3. WHAT HAPPENS WITHOUT OVERLAP

### Scenario: No Overlap (chunks of 512)

```
Window 1: tokens[0:512]
├─ Token 0: has context from tokens 1-511 (✓ good)
├─ Token 256: has context from tokens 0-511 (✓ excellent)
└─ Token 511: has context from tokens 0-510 (✗ missing right context!)

Window 2: tokens[512:1024]
├─ Token 512: has NO context from window 1 (✗ missing left context!)
├─ Token 768: has context from tokens 512-1023 (✓ good)
└─ Token 1023: has context from tokens 512-1022 (✗ missing right context!)
```

### The Problem: Edge Effects

**Tokens 0-511 get correct context for their position within their window:**
- But if an isnad spans tokens 500-520, the BERT model sees:
  - Token 500: in context [0:512] ✓
  - Token 511: in context [0:512], edge case (no tokens 512+)
  - Token 512: in context [512:1024], **NO context from tokens 0-511** ✗
  - Token 520: in context [512:1024] ✓

**Result**: Token 511 and 512 might be predicted differently even though they're part of the same isnad, because:
- Token 511 has full left+right context
- Token 512 has zero left context

### Specific Risk for Our Task

We're detecting boundary tokens (start of khabars). If a boundary spans the window edge:
- Word "أخبرنا" at token 510-512 (3 tokens)
- Token 510, 511: predicted in context [0:512] → prob=0.97 ✓
- Token 512: predicted in context [512:1024] → prob=0.45 ✗ (different!)

We'd get **inconsistent predictions for the same isnad**.

---

## 4. WHY OVERLAP HELPS

### Scenario: 50% Overlap (256 tokens overlap)

```
Window 1: tokens[0:512]
Window 2: tokens[256:768]      ← 256 tokens shared
Window 3: tokens[512:1024]     ← 256 tokens shared
...
```

### What This Does

Problematic word "أخبرنا" at tokens 510-512:

```
In Window 1 [0:512]:
  Token 510: context [0:512] → pred=0.97 ✓

In Window 2 [256:768]:
  Token 510: context [256:768] → pred=0.96 ✓
  Token 511: context [256:768] → pred=0.96 ✓
  Token 512: context [256:768] → pred=0.97 ✓
```

**Now we can reconcile**:
- Token 510-512 are predicted in multiple windows
- We can take the **max probability** across windows
- Or take the **consensus** (most windows agree)

---

## 5. THE REAL ISSUE: CURRENT IMPLEMENTATION

Looking at how the current pipeline handles overlap:

### Current Approach (from camelbert_char_boundaries_v2.json)

```python
# From convert_boundary_tokens_direct.py
for tok, off, pred, prob in zip(tokens, offsets, preds, probs):
    if tok in SPECIAL:
        continue
    cs, ce = off
    if cs == 0 and ce == 0:
        continue
    
    # Deduplication: keep max probability
    if cs not in char_to_prob or prob > char_to_prob[cs]:
        char_to_prob[cs] = prob
        char_to_pred[cs] = pred
```

**What this does:**
1. For each (token, offset, prediction, probability):
2. If `char_start` already seen, keep prediction with higher probability
3. This implicitly resolves conflicts from overlapping windows

**The Problem:**
- ✓ Handles duplicate detections (same char_start in multiple windows)
- ✗ Throws away information about **why** the probabilities differ
- ✗ Doesn't validate if the difference is meaningful or just noise
- ✗ No way to detect "this token was uncertain across windows"

---

## 6. WHAT'S THE BEST APPROACH?

### Option A: No Overlap (Aggressive)

**Pros:**
- 50% fewer forward passes
- Simpler to implement
- No conflict resolution needed
- Half the memory/compute time

**Cons:**
- Edge effects at window boundaries
- Inconsistent predictions for boundary-spanning tokens
- Requires validation/cleanup after

**Best for**: Fast iteration, preliminary screening

**Implementation:**
```python
# Process without overlap
for i in range(0, len(tokens), 512):
    window = tokens[i:i+512]
    preds = model(window)
    # Map back to original positions
    predictions[i:i+512] = preds
```

---

### Option B: Overlap with Confidence Validation (Recommended)

**Pros:**
- Consistent predictions at window boundaries
- Can detect uncertain tokens
- Higher quality inference

**Cons:**
- 2× compute time (50% overlap)
- Need conflict resolution

**Implementation:**
```python
# Process WITH overlap
stride = 256  # 50% overlap
for i in range(0, len(tokens), stride):
    window = tokens[i:min(i+512, len(tokens))]
    if len(window) < 100:  # Skip final small chunks
        break
    preds = model(window)
    
    # Store all predictions with window info
    for j, pred in enumerate(preds):
        global_pos = i + j
        if global_pos not in predictions:
            predictions[global_pos] = []
        predictions[global_pos].append(pred)

# Reconcile: for positions with multiple predictions
final_preds = {}
confidence = {}
for pos, preds_list in predictions.items():
    if len(preds_list) == 1:
        final_preds[pos] = preds_list[0]['pred']
        confidence[pos] = preds_list[0]['prob']
    else:
        # Multiple windows saw this token
        best = max(preds_list, key=lambda x: x['prob'])
        final_preds[pos] = best['pred']
        confidence[pos] = best['prob']
        # Flag if inconsistent
        if not all(p['pred'] == best['pred'] for p in preds_list):
            confidence[pos] *= 0.9  # Penalize confidence
```

---

### Option C: Smart Windowing (Optimal)

**Core Insight**: Use overlap, but only rely on the "confident center" of each window.

```python
stride = 256
margin = 64  # Ignore first/last 64 tokens of each window

for i in range(0, len(tokens), stride):
    window = tokens[i:i+512]
    preds = model(window)
    
    # Only keep predictions from "confident center"
    # Position 64 to 448 (= tokens 64-448 of the window)
    for j in range(margin, min(len(preds), 512-margin)):
        global_pos = i + j
        
        # This position has context from [i:i+512]
        # So it has context from [global_pos-margin : global_pos+margin]
        # Which is good enough
        predictions[global_pos] = preds[j]
```

**Why this works:**
- ✓ Tokens in center of window have full context (512 tokens)
- ✓ We ignore edge tokens (unreliable)
- ✓ No overlap needed for final predictions
- ✓ Clean, simple, no conflict resolution

**Trade-off:**
- Ignore 64+64=128 tokens per window
- With 256 stride, we lose coverage at very end
- Solution: Process final chunk with overlap if needed

---

## 7. TESTING: OVERLAP vs NO-OVERLAP vs SMART WINDOW

Let me analyze the ACTUAL raw_inference.json to see what happened:

### Analysis of Current Data

```
Raw inference file: 297,984 tokens
Unique char_start positions: 69,031
Boundary tokens (pred=1): 16,165

This suggests:
- Multiple tokens map to same char_start (from overlapping windows)
- Current implementation deduplicates by keeping max probability
- No information about window boundaries or confidence variance
```

**Question**: Did the Colab notebook use overlap? Let's check the actual behavior:

If no overlap was used:
- 297,984 tokens ÷ 512 = 582 windows needed
- Tokens would align perfectly with window boundaries

If 50% overlap was used:
- 297,984 tokens with 256 stride = 1,164 windows
- Same tokens appear in multiple windows
- Deduplication via max probability

**Evidence**: The large number of unique positions (69,031) vs boundary tokens (16,165) suggests overlapping windows with deduplication.

---

## 8. BEST WORKFLOW: MINIMAL POST-PROCESSING

### Recommended: Smart Window Approach

```python
# 1. INFERENCE (Colab notebook)
def infer_with_smart_windows(text, model, window_size=512, stride=256, margin=64):
    """
    Infer on text with smart windowing.
    
    Strategy: Use overlap, but keep only confident center of each window.
    """
    tokens = tokenize(text)  # Get all tokens with positions
    predictions = {}
    
    for i in range(0, len(tokens), stride):
        window_end = min(i + window_size, len(tokens))
        
        if window_end - i < 100:  # Skip tiny final chunks
            break
        
        # Run model on this window
        window_tokens = tokens[i:window_end]
        preds, probs = model(window_tokens)
        
        # Keep only confident center (margin to window_size-margin)
        safe_start = margin
        safe_end = min(window_size - margin, len(preds))
        
        for j in range(safe_start, safe_end):
            global_idx = i + j
            predictions[global_idx] = {
                'pred': preds[j],
                'prob': probs[j],
                'token': window_tokens[j]
            }
    
    # Handle final chunk if not covered
    if len(tokens) - (len(tokens) // stride) * stride > margin:
        # Process final tokens with full window
        final_start = max(len(tokens) - 512, 0)
        window_tokens = tokens[final_start:]
        preds, probs = model(window_tokens)
        for j, (pred, prob) in enumerate(zip(preds, probs)):
            global_idx = final_start + j
            if global_idx not in predictions:  # Only add if not already seen
                predictions[global_idx] = {
                    'pred': pred,
                    'prob': prob,
                    'token': window_tokens[j]
                }
    
    return predictions

# 2. MINIMAL POST-PROCESSING
def extract_boundaries_minimal(predictions, corpus_text):
    """
    Extract boundaries with minimal post-processing.
    
    Input: predictions dict (token_idx → {'pred', 'prob', 'token'})
    Output: list of boundary char positions
    """
    
    # Step 1: Filter boundary predictions only
    boundaries = [
        (idx, pred['prob'], pred['token'])
        for idx, pred in predictions.items()
        if pred['pred'] == 1
    ]
    
    # Step 2: Convert token indices to char positions using offsets
    # (This requires tokenizer to have offset_mapping)
    boundaries_chars = [
        (char_pos, prob, token)
        for token_idx, prob, token in boundaries
        # ... convert token_idx to char_pos using offsets
    ]
    
    # Step 3: Simple clustering (same as before)
    clusters = cluster_boundaries(boundaries_chars, gap=20)
    
    # Step 4: Extract cluster starts as khabar boundaries
    khabar_bounds = [c['char_start'] for c in clusters]
    
    return khabar_bounds
```

---

## 9. KEY DIFFERENCES: CURRENT VS OPTIMAL

| Aspect | Current | Optimal |
|--------|---------|---------|
| **Overlap** | 50% overlap (256 tokens) | Smart: use overlap + keep only center |
| **Deduplication** | Max probability | No dedup needed (no overlap in output) |
| **Token-Char Mapping** | Post-hoc calculation | During inference (offset_mapping) |
| **Confidence Info** | Single prob value | Implicit in margin strategy |
| **Post-processing** | Extract → Dedup → Map → Cluster | Extract → Map → Cluster |
| **Simplicity** | Medium | High |
| **Quality** | Good (F1=0.8579) | Likely better (fewer edge effects) |

---

## 10. RECOMMENDATIONS

### Short Term (Use Current, Optimize)

Current approach with overlap is fine, but:

```python
# Instead of just max probability, use:
if char_start not in boundary_by_char or prob > boundary_by_char[char_start][2]:
    # Also check: is this position near a window boundary?
    # If near boundary in both windows and confident in both, boost confidence
    boundary_by_char[char_start] = (token, offset, prob, window_consistency)
```

### Medium Term (Implement Smart Windows)

For next version of CAMeL-BERT inference:

1. Use 50% overlap (256 stride)
2. **Keep only tokens from position 64 to 448** (center 384 tokens)
3. No deduplication needed
4. Direct token→char mapping
5. Straight to clustering

**Expected benefit**: Cleaner inference, fewer edge artifacts, simpler pipeline.

### Long Term (Experiment Both)

Test on Kitab Uqala:
- Version A: Current (overlap + dedup + max prob)
- Version B: Smart windows (overlap + center only)
- Compare F1 scores

---

## 11. CONCLUSION

**Your question was spot-on.** The current approach with overlap + deduplication is a workaround for edge effects. A better approach exists:

✅ **Use overlap to get better context**  
✅ **But ignore unreliable edges** (keep only center margin)  
✅ **This eliminates the need for deduplication**  
✅ **Result: cleaner inference pipeline**

The current pipeline **works fine** (F1=0.8579), but it's doing unnecessary work. With smart windowing, you could likely improve to **F1 ≈ 0.87-0.88** with the same model.

Would you like me to:
1. Implement the smart window strategy?
2. Compare it empirically against current approach?
3. Create an improved Colab notebook with this approach?

