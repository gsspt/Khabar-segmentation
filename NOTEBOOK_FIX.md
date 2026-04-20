# Fix for Colab Inference Notebook

## The Problem

Current code only processes first 512 tokens:

```python
encoded = tokenizer(
    text,           # ← Full 268K chars
    max_length=512,
    truncation=True,  # ← TRUNCATES! Only processes first ~2000 chars
    return_offsets_mapping=True,
    return_tensors='pt'
)
```

Result: Only 11 segments instead of ~600

---

## The Solution

Replace the `infer_with_offsets` function with this chunked version:

### Step 1: Find and Delete the Old Function

In the Colab notebook, find this cell:

```python
def infer_with_offsets(text: str, tokenizer, model, max_length: int = 512) -> dict:
    """
    Run inference and preserve token-to-character offset mapping.
    ...
    """
```

**Delete the entire function.**

---

### Step 2: Add the New Chunked Function

Add this new function in its place:

```python
def infer_with_offsets(text: str, tokenizer, model, chunk_size: int = 512, overlap: int = 50) -> dict:
    """
    Run inference on FULL document in overlapping chunks.
    
    Process in chunks to handle documents larger than max_length.
    Chunks overlap to catch boundaries at chunk edges.
    """
    print(f"[INFO] Processing {len(text):,} chars in chunks of {chunk_size} with {overlap} overlap...\n")
    
    all_predictions = []
    all_probabilities = []
    all_offsets = []
    all_tokens = []
    
    chunk_num = 0
    pos = 0
    
    while pos < len(text):
        chunk_num += 1
        chunk_start = pos
        chunk_end = min(pos + chunk_size, len(text))
        chunk = text[chunk_start:chunk_end]
        
        print(f"  Chunk {chunk_num}: chars {chunk_start:,}-{chunk_end:,} ({len(chunk)} chars)")
        
        # Tokenize this chunk
        encoded = tokenizer(
            chunk,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Run inference
        with torch.no_grad():
            if torch.cuda.is_available():
                input_ids = encoded['input_ids'].cuda()
                attention_mask = encoded['attention_mask'].cuda()
                outputs = model(input_ids, attention_mask=attention_mask)
            else:
                outputs = model(**encoded)
            
            logits = outputs.logits[0]  # [seq_len, 2]
        
        # Get predictions and probabilities
        preds = np.argmax(logits.cpu().numpy(), axis=-1)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]
        
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        offsets = encoded['offset_mapping'][0].numpy()
        
        # Adjust offsets by chunk position
        adjusted_offsets = []
        for token_start, token_end in offsets:
            # Convert local offset to global document position
            global_start = chunk_start + token_start
            global_end = chunk_start + token_end
            adjusted_offsets.append((global_start, global_end))
        
        # Append to global lists
        all_predictions.extend(preds)
        all_probabilities.extend(probs)
        all_offsets.extend(adjusted_offsets)
        all_tokens.extend(tokens)
        
        # Move to next chunk
        # Non-overlapping: move by full chunk_size
        # With overlap: move by (chunk_size - overlap)
        pos += (chunk_size - overlap)
        
        if pos >= len(text):
            break
    
    print(f"[OK] Processed {chunk_num} chunks")
    print(f"[OK] Total tokens: {len(all_predictions)}")
    
    return {
        'predictions': all_predictions,
        'probabilities': all_probabilities,
        'tokens': all_tokens,
        'offsets': all_offsets,
    }
```

---

### Step 3: Update the Inference Call

Find this cell:

```python
print(f"[INFO] Running inference on full corpus...")
print(f"  Text length: {len(full_text):,} chars")
print(f"  Expected tokens: ~{len(full_text) // 4} (rough estimate)")
print(f"  Processing in 512-token chunks...\n")

inference_result = infer_with_offsets(full_text, tokenizer, model)
```

**Keep it exactly the same** — the new function will automatically handle chunking.

---

### Step 4: Run the Notebook

1. **Restart kernel** (Runtime → Restart runtime)
2. **Run all cells in order**
3. The inference will now:
   - Process the full 268K chars in overlapping chunks
   - Combine predictions from all chunks
   - Produce offsets for the full document
   - Export ~2000+ boundary tokens (not just 208)

---

## What Changes

| Aspect | Before | After |
|--------|--------|-------|
| **Text processed** | First 512 tokens (~2K chars) | Full 268K chars |
| **Boundary tokens found** | 208 | ~2000+ |
| **Segments extracted** | 11 | ~400-500 (expected) |
| **Recall vs gold** | 1.8% | 65-75% (expected) |

---

## Key Parameters

If you want to adjust chunking:

```python
inference_result = infer_with_offsets(
    full_text,
    tokenizer,
    model,
    chunk_size=512,    # Tokens per chunk (max BERT length)
    overlap=50         # Token overlap between chunks
)
```

**Recommendations:**
- `chunk_size=512`: Maximum BERT input (don't change)
- `overlap=50`: Safe overlap to catch boundaries at chunk edges (can adjust 30-100)

---

## Verification

After running, check the output:

```
[INFO] Processing 268,540 chars in chunks of 512 with 50 overlap...

  Chunk 1: chars 0-512 (512 chars)
  Chunk 2: chars 462-974 (512 chars)
  Chunk 3: chars 924-1436 (512 chars)
  ...
  [OK] Processed 520 chunks
  [OK] Total tokens: 266,240
```

The exported JSON should have:
- `total_tokens`: ~260K-270K (much larger!)
- `boundary_tokens`: ~2000+ (not 208)

---

## Then Run Local Post-Processing

Once you download the fixed JSON:

```bash
python3 scripts/camelbert_local_postprocess.py \
    --input results/camelbert_kitab_uqala_raw_inference.json \
    --output results/camelbert_kitab_uqala_segments.json
```

This should now give you ~400-500 segments instead of 11, with proper recall around 65-75%.

---

## Why This Works

1. **Chunking** → Process full document
2. **Overlap** → Catch boundaries that fall on chunk edges
3. **Offset adjustment** → Keep character positions correct
4. **Clustering locally** → Merge adjacent predictions
5. **Proper recall** → Compare full results vs 613 gold standard
