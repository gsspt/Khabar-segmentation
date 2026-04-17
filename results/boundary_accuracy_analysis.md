# Analyse de la Précision des Frontières

## Configuration
- Corpus: Reference corpus (kitab_uqala) - 268,540 chars
- Boundaries de référence: 613
- Baseline: Améliorée (verbes fiables uniquement, strict=True)

## Résultats de Détection
- Akhbars détectés: 651
- Boundaries matchés: 178 (27.3%)
- Non matchés (détectés): 473
- Non matchés (référence): 461 (75.2%)

## Précision des Frontières

### Erreurs de Position (START)
- Moyenne: 109.2 chars
- Médiane: 89.0 chars
- Max: 465 chars
- Exact: 9/178 (5.1%)
- ±10 chars: 13/178 (7.3%)

### Erreurs de Position (END)
- Moyenne: -134.6 chars
- Médiane: -108.0 chars
- Max: 459 chars
- Exact: 22/178 (12.4%)
- ±10 chars: 22/178 (12.4%)

### Erreurs de Longueur
- Moyenne: -243.9 chars
- Médiane: -239.0 chars
- Max: 498 chars

## Catégorisation des Erreurs
- Matches exacts: 2 (1.1%)
- Petites erreurs (±10): 2 (1.1%)
- Erreurs moyennes (±50): 3 (1.7%)
- Grandes erreurs (>50): 173 (97.2%)

## Verdict
[FAIBLE] La baseline a des erreurs importantes de positionnement
