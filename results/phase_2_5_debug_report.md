# Phase 2.5: Debug Report - Undetected Verb Analysis

## Key Finding

57% of undetected akhbars have verbs ALREADY in v3 list.
This indicates **boundary detection failure**, not missing verbs.

## Most Common Undetected Verb

- Count: 1 undetected instances
- % of undetected: 0.2%
- Is in v3 verb list: YES
- Avg segment length: 567 chars
- Has qal marker: 144/1 (14400%)

## Failure Analysis

| Failure Mode | Count |
|--------------|-------|
| verb_not_found | 5 |
| length_validation | 1 |

## Potential Fixes

1. **Length validation**: May be too strict
2. **Consider**: Different bounds for different verb types

