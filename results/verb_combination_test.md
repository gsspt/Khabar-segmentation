# Verb Combination Testing Results

Testing different verb sets to optimize detection.

| Config | Verb Count | Detected | % | Improvement |
|--------|------------|----------|---|-------------|
| Base + Prefixed (Current V3) | 65 | 416/613 | 67.9% | +20 |
| Base + Prefixed + قال | 66 | 81/613 | 13.2% | -315 |
| Base + Prefixed + ومن | 66 | 452/613 | 73.7% | +56 |
| Base + Prefixed + وله | 66 | 419/613 | 68.4% | +23 |
| Base + Prefixed + all missing | 68 | 95/613 | 15.5% | -301 |

## Findings

- Baseline V2: 396 detected
- Best: Base + Prefixed + ومن = 452 detected
- Improvement: +56 akhbars
