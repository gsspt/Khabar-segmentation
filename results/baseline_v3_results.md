# Baseline V3: Refined ISNAD-First Approach

## Stratégie (v3 improvements)
1. Dynamic isnad end boundary (use next isnad as upper bound)
2. Expanded ISNAD_START_VERBS (prefixed variants + missing high-frequency verbs)
3. Relaxed length validation (allow longer isnads, safety cap only)
4. Fallback boundary detection for isnads without 'قال'
5. Strict word boundaries for common words (قال, ومن, وله)

## Résultats
- Akhbars détectés: 452
- Akhbars attendus: 613
- Écart: -161 (-26.3%)

## Statistiques
- Couverture: 65.7%
- Longueur isnad moyenne: 78 chars
- Longueur khabar moyenne: 313 chars

## Verdict
[ACCEPTABLE] Écart modéré
