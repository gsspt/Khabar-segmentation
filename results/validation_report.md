# Rapport de Validation : Baseline vs Annotations Manuelles

## Configuration
- **Annotations** : 612 akhbars du Kitab Uqala al-Majanin
- **Texte brut** : 187,547 caractères (406IbnHabibNaysaburi)
- **Méthode** : Fuzzy matching (threshold=0.82) + segeval metrics

## Résultats de Matching
- **Akhbars trouvés** : 0 / 612
- **Couverture** : 0.0%

## Métriques de Segmentation

[!]  Erreur segeval: 

### F1 sur Frontières (tolérance ±50 chars)
- **F1 moyenne** : 0.0000
- **F1 début de segment** : 0.0000
  - Précision : 0.0000 (0/0)
  - Rappel : 0.0000
- **F1 fin de segment** : 0.0000
  - Précision : 0.0000 (0/0)
  - Rappel : 0.0000

## Résumé
- **Référence (annotations)** : 0 akhbars
- **Hypothèse (baseline)** : 0 akhbars
- **Différence** : N/A (aucun akhbar trouv­ dans annotations)

**Verdict** : [ERR] FAIBLE

## Notes
- Le fuzzy matching a une tolérance de 0.82 (82% de similarité)
- La F1 sur frontières accepte une tolérance de ±50 caractères
- Les akhbars très courts ou mal trouvés peuvent affecter les métriques
