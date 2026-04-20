# Khabar Segmentation Pipeline

## 📋 Vue d'ensemble

Ce projet compare trois approches de segmentation de textes arabes historiques en unités narratives (akhbars) :

1. **Baseline v4** : Rule-based linguistique
2. **CAMeL-BERT** : Fine-tuned token classification
3. **Gold Standard** : Vérité de référence générée par Deepseek API

---

## 🗂️ Structure du Repo

```
scripts/
├── baselines/
│   └── baseline_v4.py           # Approche rule-based (isnad detection + boundaries)
│
├── finetuning/
│   ├── prepare_binary_classification_data_v2.py
│   └── prepare_binary_classification_data.py
│
├── gold_standard/
│   └── create_gold_standard_deepseek.py  # Génère la vérité de référence
│
├── evaluation/
│   └── compare_baselines.py     # Compare les trois approches
│
└── archive/                     # Anciens scripts et expériences
```

**Notebooks Colab** :
- `notebooks/camelbert_binary_classification_finetuning.ipynb` - Fine-tuning
- `notebooks/camelbert_binary_classification_inference.ipynb` - Inférence
- `notebooks/camelbert_test_openiti_generalization.ipynb` - Test généralisation

---

## 🚀 Pipeline d'Exécution

### Phase 1 : Créer le Gold Standard

```bash
# Générer les annotations Deepseek
python3 scripts/gold_standard/create_gold_standard_deepseek.py

# Output:
# → data/gold_standard/gold_standard_0392IbnIsmacil.json
```

**Format du Gold Standard** :
```json
{
  "akhbars": [
    {
      "id": 1,
      "type": "with_isnad",              // with_isnad | poetry | prose | continuation
      "isnad": "حدثنا محمد",             // (seulement si type=with_isnad)
      "content": "قال رأيت النبي",       // texte du matn ou du contenu
      "confidence": 0.95,                // 0.0-1.0
      "notes": "إسناد واضح"              // optionnel
    }
  ]
}
```

### Phase 2 : Tester la Baseline v4

```bash
python3 scripts/baselines/baseline_v4.py \
  --input openiti_corpus/data/0392IbnIsmacilMisri/... \
  --output results/baseline_v4_results.json
```

### Phase 3 : Fine-tune CAMeL-BERT (Colab)

1. Lancer le notebook : `notebooks/camelbert_binary_classification_finetuning.ipynb`
2. Résultats sauvegardés : `checkpoints/camelbert_binary_classification_final/`

### Phase 4 : Inférence CAMeL-BERT (Colab)

1. Lancer : `notebooks/camelbert_binary_classification_inference.ipynb`
2. Tester sur le même corpus que baseline_v4

### Phase 5 : Comparer les Résultats

```bash
python3 scripts/evaluation/compare_baselines.py

# Output:
# - Métriques de détection (recall, precision, F1)
# - Confiance des prédictions
# - Exactitude des boundaries
# - Rapport de comparaison
```

---

## 📊 Métriques d'Évaluation

### Gold Standard

- **Total akhbars** : N segments annotés par Deepseek
- **Confiance moyenne** : Score de certitude de Deepseek (0-1)
- **Distribution par type** :
  - `with_isnad` : segments avec chaîne de transmission
  - `poetry` : passages poétiques
  - `prose` : textes narratifs
  - `continuation` : continuations du segment précédent

### Baseline v4

- **Détection** : % d'akhbars trouvés vs gold standard
- **Boundaries** : Exactitude des positions (chars/tokens)
- **Faux positifs** : Segments mal détectés
- **Temps d'inférence** : Rapidité relative

### CAMeL-BERT

- **Accuracy** : % de tokens correctement classifiés
- **F1 Score** : Moyenne harmonique precision/recall
- **Confiance** : Probabilités moyennes par token
- **Généralisation** : Performance sur corpus externe (OpenITI)

### Comparaison

| Métrique | Baseline v4 | CAMeL-BERT | Gold Standard |
|----------|-------------|------------|---------------|
| Recall | ? | ? | 1.0 (référence) |
| Precision | ? | ? | N/A |
| F1 | ? | ? | N/A |
| Speed | Fast | Slow | N/A |
| Confiance | Binary | 0-1 | 0-1 |

---

## 🔧 Configuration

### Variables d'environnement (.env)

```bash
# Deepseek API
DEEPSEEK_API_KEY=sk_...

# Optionnel : HuggingFace (pour CAMeL-BERT)
HF_TOKEN=hf_...
```

### Corpus OpenITI

Texte court (4.1K mots) pour tests rapides :
```
openiti_corpus/data/0392IbnIsmacilMisri/
  └── 0392IbnIsmacilMisri.CuqalaMajanin/
      └── 0392IbnIsmacilMisri.CuqalaMajanin.Shamela0027093-ara1
```

---

## 📈 Résultats Attendus

### CAMeL-BERT (test set)
- **Accuracy** : 99.79%
- **F1 Score** : 98.11%
- Excellent performance sur données d'entraînement

### Généralization
- À tester sur corpus OpenITI external
- Vérifier la robustesse cross-domain

### Baseline v4
- Performance rapide
- Bonne baseline pour comparaison

---

## 📝 Notes Importantes

1. **Deepseek API** : Consomme des crédits API. Tester d'abord sur petit corpus.
2. **CAMeL-BERT** : Nécessite GPU pour entraînement. Colab recommandé.
3. **Gold Standard** : Pas parfait - annotations créées par LLM, vérifier manuellement.
4. **Format unifiéalisé** : Tous les résultats doivent être sauvegardés dans `data/gold_standard/` ou `results/`.

---

## 🚧 Checklist

- [ ] Gold standard généré (Deepseek)
- [ ] Baseline v4 testé sur même corpus
- [ ] CAMeL-BERT fine-tuné et évalué
- [ ] Comparaison des trois approches
- [ ] Rapport final généré
- [ ] Généralisation sur OpenITI validée

---

## 📚 Références

- CLAUDE.md : Documentation projet globale
- BASELINE_RESULTS.md : Résultats détaillés baseline v4
- notebooks/ : Scripts Colab interactifs
