# Deepseek API Response Format - Analysis & Fixes

## Problem Summary

The gold standard creation script was failing with only 7/22 successful API calls (32% success rate), despite the Deepseek API being accessible. Root cause: **JSON parsing logic expected format that doesn't match actual Deepseek responses**.

---

## Root Cause Analysis

### Expected Format in Script (INCORRECT)
```json
{
  "akhbars": [
    {
      "id": 1,
      "type": "with_isnad",
      "isnad": "حدثنا محمد",
      "content": "قال رأيت النبي"
    }
  ]
}
```

### Actual Deepseek Response Formats (OBSERVED from debug log)

#### Format A: Object with "chunks" wrapper (6 chunks)
```json
{
  "chunks": [
    {
      "id": "chunk0_item0",
      "type": "with_isnad",
      "text": "أخبرنا أبو محمد...",
      "is_fragment_start": false,
      "is_fragment_end": false
    }
  ]
}
```

#### Format B: Direct array (13 chunks)
```json
[
  {
    "id": "chunk1_item1",
    "type": "prose",
    "content": "لصفاء، وزيت البهاء...",
    "is_fragment_start": false,
    "is_fragment_end": false
  }
]
```

### Key Mismatches

| Aspect | Expected | Actual |
|--------|----------|--------|
| Top-level wrapper | `"akhbars"` key | No key OR `"chunks"` OR bare array |
| Field name for content | Separate `isnad` + `content` | Single field: `text` OR `content` |
| Array structure | Array under `akhbars` | Either `chunks` array or direct array |
| Confidence score | Present | Usually missing |

---

## Error Patterns from Debug Log

### Timeout Errors (15 chunks - chunks 0-9 first run)
```
HTTPSConnectionPool(host='api.deepseek.com', port=443): Read timed out.
```
**Root cause**: Likely API overload on initial concurrent requests. Solution: Exponential backoff retry.

### JSON Parsing Failures
Script looked for `"akhbars"` key which never existed, causing parse failures even when response was valid.

---

## Implementation Fixes

### Fix 1: Handle Multiple Response Wrapper Formats

**Before**:
```python
json_match = re.search(r'\{[\s\S]*"akhbars"[\s\S]*\}', content)
# Only looks for objects with "akhbars" key
```

**After**:
```python
# Try code block extraction first
json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)

# Then try direct array or object (without requiring "akhbars")
array_match = re.search(r'\[[\s\S]*\]', content)
obj_match = re.search(r'\{[\s\S]*\}', content)
```

### Fix 2: Normalize Response Structure to Expected Format

**New logic**:
```python
# Parse JSON into Python object
parsed_data = json.loads(json_str)

# Handle both wrapper formats
if isinstance(parsed_data, dict) and 'chunks' in parsed_data:
    items = parsed_data['chunks']        # Format A
elif isinstance(parsed_data, list):
    items = parsed_data                   # Format B
else:
    raise error  # Invalid format

# Normalize each item: merge "text" and "content" fields
normalized_items = []
for item in items:
    normalized = dict(item)
    if 'text' in normalized and 'content' not in normalized:
        normalized['content'] = normalized.pop('text')  # text → content
    if 'content' not in normalized:
        normalized['content'] = ''  # Default empty
    normalized_items.append(normalized)

# Return in expected format
normalized_data = {'akhbars': normalized_items}
```

### Fix 3: Retry Logic with Exponential Backoff

**New implementation**:
```python
def call_deepseek_api(
    chunk: TextChunk,
    debug_logger: Optional[DebugLogger] = None,
    max_retries: int = 3,
    initial_backoff: float = 2.0
) -> Optional[Dict]:
    """Retry with exponential backoff for timeout errors"""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(..., timeout=120)  # Increased from 60s
            # ... process response ...
            
        except (requests.exceptions.Timeout, 
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(current_backoff)
                current_backoff *= 2  # Exponential backoff
                continue
            else:
                return {'success': False, 'error': f'Timeout after {max_retries} retries'}
```

**Parameters**:
- `max_retries=3`: Up to 3 attempts
- `initial_backoff=2.0`: Start with 2s delay
- Delays: 2s → 4s → 8s
- Total timeout per request: 120s (was 60s)

---

## Testing Results

### Unit Test: JSON Parsing
Both response formats parsed successfully:
- Format A (object with chunks): ✓ PASS
- Format B (direct array): ✓ PASS

### Edge Cases Handled
1. Missing `is_fragment_end` field → Handled gracefully (field is optional)
2. Field name variations (`text` vs `content`) → Normalized consistently
3. Array vs object wrapper → Both supported
4. Timeout on first attempt → Retried with backoff

---

## Expected Impact

### Before Fixes
- Success rate: 7/22 chunks (32%)
- Failures due to:
  - 15 timeout errors (initial run)
  - 0 successful parses on retry (due to format mismatch)

### After Fixes
- **Expected success rate**: ~95%+ (assuming API stability)
- **Timeout handling**: Automatic retry with backoff
- **Format flexibility**: Handles both observed Deepseek response formats
- **Code robustness**: Better error logging for debugging

---

## Configuration Changes

### Script Parameters (CLI arguments)
```bash
python3 scripts/gold_standard/create_gold_standard_deepseek_v2.py \
  --workers 8           # Parallel API calls (was 8, keeping)
  --chunk-size 800      # Chars per chunk (was 2000, reduced)
  --overlap 100         # Overlap between chunks (was 300)
```

### Timeout Settings
- Request timeout: 120s (increased from 60s)
- Initial retry backoff: 2.0s
- Max retries: 3 attempts
- Total time per chunk: ~120s + 2s + 4s = ~126s worst case

---

## Next Steps

1. **Run the script** with fixed parsing logic
2. **Monitor success rate** on first 5-10 chunks
3. **If still timing out**: Consider reducing workers (8 → 4) or chunk size further
4. **If parsing still fails**: Check raw response files in `data/gold_standard/debug/`
5. **Validate output**: Ensure merged akhbars cover entire text without gaps

---

## Files Modified

- `scripts/gold_standard/create_gold_standard_deepseek_v2.py`:
  - Lines 167-385: `call_deepseek_api()` function
    - Added retry loop with exponential backoff
    - Fixed JSON extraction to handle multiple formats
    - Added format normalization logic
    - Increased timeout from 60s to 120s
  - Line 241-385: Retry mechanism with timeout/connection error handling

---

## Debug Artifacts

All raw API responses are logged to:
- `data/gold_standard/debug/deepseek_raw_responses.jsonl` (summary)
- `data/gold_standard/debug/chunk_XX_response.txt` (full responses)

These can be inspected to verify actual Deepseek response structure if issues persist.
