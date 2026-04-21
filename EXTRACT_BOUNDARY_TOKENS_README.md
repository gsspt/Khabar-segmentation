# Extraction des Boundary Tokens - CAMeL-BERT

**Objectif:** Extraire les tokens individuels prédits comme **isnads** (boundary tokens) par le modèle CAMeL-BERT du corpus Kitab Uqala.

## 📋 Fichiers Fournis

### 1. **Script Python** (Local ou Colab)
- **Chemin:** `scripts/extract_boundary_tokens.py`
- **Utilisation:** Peut être exécuté sur Colab ou dans un environnement avec PyTorch
- **Commande:**
  ```bash
  python3 scripts/extract_boundary_tokens.py
  ```

### 2. **Notebook Jupyter** (Google Colab)
- **Chemin:** `notebooks/extract_boundary_tokens_colab.ipynb`
- **Utilisation recommandée:** Google Colab (pour facilité d'exécution)
- **Avantages:** Exécution cellule par cellule, prévisualisation, téléchargement direct

---

## 🚀 Comment Utiliser

### Option A : Google Colab (Recommandée)

1. **Ouvrir le notebook Colab:**
   - Aller à [Google Colab](https://colab.research.google.com)
   - Menu `File` → `Open notebook`
   - Coller le lien GitHub ou importer le fichier `notebooks/extract_boundary_tokens_colab.ipynb`

2. **Cloner le repository (si nécessaire):**
   ```bash
   !git clone https://github.com/YOUR_REPO/Khabar-segmentation.git
   %cd Khabar-segmentation
   ```

3. **Exécuter les cellules dans l'ordre:**
   - Installation des dépendances
   - Chargement du tokenizer
   - Chargement du corpus
   - Extraction des boundary tokens
   - Téléchargement des résultats

### Option B : Python Local (Requiert PyTorch)

```bash
cd /path/to/Khabar-segmentation
python3 scripts/extract_boundary_tokens.py
```

⚠️ **Note:** PyTorch doit être installé (ce qui n'est pas le cas dans l'environnement Windows actuel)

---

## 📊 Résultat Attendu

### Fichier Généré
- **Nom:** `results/camelbert_boundary_tokens_clean.json`
- **Taille:** ~800 KB
- **Format:** JSON structuré

### Contenu

```json
{
  "metadata": {
    "corpus": "data/processed/kitab_uqala_reference_corpus.txt",
    "corpus_size_chars": 268540,
    "corpus_size_tokens": 297984,
    "model": "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    "total_boundary_tokens": 17938,
    "boundary_percentage": 6.03
  },
  "statistics": {
    "total_tokens": 297984,
    "boundary_tokens_count": 17938,
    "non_boundary_tokens": 280046,
    "boundary_ratio": 0.0603
  },
  "boundary_tokens": [
    "[CLS]",
    "أبو",
    "بكر",
    "محمد",
    "بن",
    "عبدالله",
    ...
  ],
  "boundary_indices": [0, 583, 590, 591, 592, 593, ...]
}
```

---

## 🔍 Explication

### Qu'est-ce qu'un Boundary Token?

Un **boundary token** est un token individuel que le modèle CAMeL-BERT a classifié comme faisant partie d'un **isnad** (chaîne de transmission).

- **Prediction = 1** → Token isnad (boundary)
- **Prediction = 0** → Token prose (non-boundary)

### Structure du Fichier

| Clé | Contenu |
|-----|---------|
| `metadata` | Informations sur le corpus et le modèle |
| `statistics` | Statistiques agrégées |
| `boundary_tokens` | Liste simple des 17,938 tokens prédits comme isnads |
| `boundary_indices` | Index de position dans le corpus tokenisé |

### Utilisation

#### Pour une Analyse Simple
```python
import json

with open('results/camelbert_boundary_tokens_clean.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Liste des tokens
boundary_tokens = data['boundary_tokens']
print(f"Total: {len(boundary_tokens)}")
print(f"Premiers 20: {boundary_tokens[:20]}")
```

#### Pour Rejouer la Segmentation
```python
# Utiliser les indices pour reconstruire le corpus annoté
tokens = data['boundary_tokens']
indices = data['boundary_indices']

# Créer un mapping index → token
index_to_token = {idx: token for idx, token in zip(indices, boundary_tokens)}

# Afficher les tokens à chaque position boundary
for idx in indices[:30]:
    print(f"Position {idx}: {index_to_token[idx]}")
```

---

## 📈 Statistiques Attendues

À partir des données existantes:

```
Corpus:
  Caractères: 268,540
  Tokens: 297,984

Boundary Tokens (Isnads):
  Count: 17,938
  Percentage: 6.03%
  Indices: [583, 590, 591, 592, ...]
```

---

## ⚙️ Détails Techniques

### Tokenizer Utilisé
- **Modèle:** CAMeL-Lab/bert-base-arabic-camelbert-msa
- **Type:** AutoTokenizer (BERT tokenizer)
- **Caractéristiques:** Tokenization par sous-mots (WordPiece), support complet UTF-8

### Format des Prédictions Source
- **Fichier:** `results/camelbert_kitab_uqala_raw_inference.json`
- **Contenu:** Array de 297,984 valeurs (0 ou 1)
- **Modèle:** CAMeL-BERT classification binaire (isnad vs prose)

---

## 🐛 Troubleshooting

### Erreur: "Module specified is not found"
**Cause:** PyTorch n'est pas installé localement  
**Solution:** Utilisez Google Colab

### Erreur: "Mismatch entre tokens et prédictions"
**Cause:** Tokenizer légèrement différent  
**Solution:** Le script utilise le minimum des deux tailles

### Fichier JSON trop volumineux
**Cause:** 17,938 tokens × ~50 caractères = ~900 KB  
**Solution:** C'est normal, réduire si nécessaire avec compression

---

## 💡 Cas d'Usage

### 1. Analyse des Isnads Détectés
```python
# Quels tokens détecte le modèle comme isnads?
boundary_tokens = data['boundary_tokens']
print(set(boundary_tokens))  # Tokens uniques
```

### 2. Visualisation des Positions
```python
# Où sont situés les isnads dans le corpus?
indices = data['boundary_indices']
print(f"Première isnad: position {indices[0]}")
print(f"Dernière isnad: position {indices[-1]}")
```

### 3. Comparaison avec Gold Standard
```python
# Comparer les positions avec les vraies boundaries
gold_standard = json.load(open('data/processed/kitab_uqala_boundaries.json'))
camelbert_indices = data['boundary_indices']

# Analyser les alignements
...
```

---

## 📝 Fichiers Connexes

- `results/camelbert_kitab_uqala_raw_inference.json` — Prédictions brutes (0/1)
- `results/camelbert_kitab_uqala_segments.json` — Segments post-processés
- `data/processed/kitab_uqala_boundaries.json` — Gold standard

---

## ✅ Checklist d'Exécution

- [ ] Cloner/télécharger le repository
- [ ] Ouvrir le notebook Colab
- [ ] Exécuter les cellules dans l'ordre
- [ ] Télécharger `camelbert_boundary_tokens_clean.json`
- [ ] Vérifier la taille (~800 KB)
- [ ] Valider le contenu (17,938 tokens)

---

**Générée:** 2026-04-21  
**Version:** 1.0
