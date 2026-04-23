# Colab Offset Accuracy Diagnostics

90% accuracy is **too low**. This indicates the offsets might still be chunk-relative or there's a tokenizer issue. 

## Quick Diagnostic (Add to Colab Notebook)

Add this **between Cell 5 and Cell 6** to identify the problem:

```python
# ===== DETAILED OFFSET DIAGNOSIS =====
print("\n=== OFFSET ACCURACY DETAILED ANALYSIS ===\n")

# Analyze all tokens (not just sample)
sample_size = min(500, len(all_tokens))
matches = 0
mismatches_by_type = {
    'length_mismatch': [],
    'character_mismatch': [],
    'whitespace_diff': [],
    'out_of_bounds': [],
}

for i in range(sample_size):
    token = all_tokens[i]
    offset = all_offsets[i]
    char_start, char_end = offset
    
    # Skip special tokens
    if token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]"]:
        continue
    
    # Check if offset is within corpus
    if not (0 <= char_start < len(text) and 0 <= char_end <= len(text)):
        mismatches_by_type['out_of_bounds'].append({
            'token': token, 'offset': offset, 'text_len': len(text)
        })
        continue
    
    # Get corpus text at offset
    corpus_at_offset = text[char_start:char_end]
    token_clean = token.lstrip('#')
    
    if corpus_at_offset == token_clean:
        matches += 1
    else:
        # Categorize mismatch
        error = {
            'token': token,
            'offset': offset,
            'corpus': repr(corpus_at_offset[:30]),
            'expected': repr(token_clean[:30])
        }
        
        if len(corpus_at_offset) != len(token_clean):
            mismatches_by_type['length_mismatch'].append(error)
        elif corpus_at_offset.replace(' ', '') == token_clean.replace(' ', ''):
            mismatches_by_type['whitespace_diff'].append(error)
        else:
            mismatches_by_type['character_mismatch'].append(error)

accuracy = 100 * matches / sample_size
print(f"Accuracy: {matches}/{sample_size} = {accuracy:.1f}%\n")

if accuracy < 95:
    print("⚠️  ACCURACY TOO LOW - Analyzing error types:\n")
    
    for error_type, errors in mismatches_by_type.items():
        if errors:
            print(f"{error_type}: {len(errors)} errors")
            if errors:
                print(f"  Example: {errors[0]}\n")
    
    print("\n📋 DIAGNOSIS HINTS:")
    
    # Check chunk overlap issue
    print("\n1. Check if offsets span chunk boundaries:")
    for i in [0, 5, 10, 200, -1]:
        if 0 <= i < len(all_offsets):
            off = all_offsets[i]
            print(f"   Token {i}: offset={off}, text_snippet='{text[off[0]:min(off[1], off[0]+20)]}'")
    
    # Check if pattern repeats every CHUNK_SIZE
    print("\n2. Check if offset errors repeat with CHUNK_SIZE pattern:")
    error_indices = []
    for i in range(min(sample_size, len(all_tokens))):
        token = all_tokens[i]
        offset = all_offsets[i]
        if not token.startswith('['):
            corpus_at = text[offset[0]:offset[1]] if 0 <= offset[0] < len(text) and 0 <= offset[1] <= len(text) else ""
            if corpus_at != token.lstrip('#'):
                error_indices.append(i)
    
    if error_indices:
        print(f"   Error indices: {error_indices[:10]}")
        gaps = [error_indices[i+1] - error_indices[i] for i in range(len(error_indices)-1)]
        if gaps:
            print(f"   Gap between errors: {gaps[:5]}")
            if all(abs(g - CHUNK_SIZE) < 10 for g in gaps):
                print("   ⚠️  PATTERN FOUND: Errors repeat every ~CHUNK_SIZE tokens")
                print("   → Offsets are likely STILL CHUNK-RELATIVE")
                print("   → FIX: Increase OVERLAP or re-check chunk offset calculation")

else:
    print("✓ Accuracy is good (>95%)")
    print("✓ Offsets are ABSOLUTE corpus positions")
    print("✓ Ready to proceed with post-processing")
```

## What This Tests

1. **Length mismatches** → Tokenizer/corpus encoding difference
2. **Character mismatches** → Whitespace/normalization issue  
3. **Out of bounds** → Offsets exceeding corpus length
4. **Repeating pattern** → Chunk overlap handling bug

## Common Issues & Fixes

### Issue 1: Offsets Still Chunk-Relative (Most Likely)
**Signal:** Errors repeat every ~500 tokens
```python
# FIX: Ensure this line is correct:
offsets_absolute = [[int(start_char + int(offset[0])), int(start_char + int(offset[1]))]
                   for offset in offsets_chunk_relative]
```

### Issue 2: Chunk Overlap Problems
**Signal:** Tokens at chunk boundaries have wrong offsets
```python
# Current: Skip 5 tokens from next chunk
skip = 5

# TRY: Skip more tokens to avoid overlap region
skip = 10  # or 20
```

### Issue 3: Tokenizer Normalization
**Signal:** Arabic text matches but with different diacritics
```python
# In Colab, check tokenizer settings:
print("Tokenizer do_lower_case:", tokenizer.do_lower_case)
print("Tokenizer do_basic_tokenize:", tokenizer.do_basic_tokenize)
```

### Issue 4: Text Encoding Mismatch
**Signal:** Random character mismatches
```python
# Verify corpus encoding:
with open(corpus_file, encoding='utf-8-sig') as f:  # Try utf-8-sig
    text = f.read()
```

## Quick Test (Local)

Once you download the JSON, run this:
```bash
python scripts/diagnose_offset_accuracy.py \
  --raw_inference results/[TEXT]/camelbert_[TEXT]_raw_inference_improved.json \
  --corpus data/processed/[TEXT]_clean.txt \
  --sample_size 500
```

## Expected Output

**Good (>95%):**
```
Accuracy: 475/500 = 95.0%
✓ Accuracy is acceptable (>95%)
✓ Offsets are ABSOLUTE corpus positions
✓ Safe to proceed with convert_boundary_tokens_direct.py
```

**Bad (<95%):**
```
Accuracy: 430/500 = 86.0%
⚠️  ACCURACY TOO LOW!

Error breakdown:
  length_mismatch                : 45 (10.5%)
  character_mismatch             : 25 (  5.8%)
  ...

DIAGNOSIS:
- Offsets may be chunk-relative, not absolute
- Check chunk overlap handling
```

## Next Steps

1. **Add diagnostic cell to Colab notebook**
2. **Re-run and check output** → identify which error type dominates
3. **Apply fix based on diagnosis**
4. **Re-validate** → should achieve >98% accuracy
5. **Then proceed** with post-processing

If you get <95% accuracy, **do not proceed with post-processing** — the results will be unreliable.
