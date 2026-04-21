# Khabar Segmentation — CLAUDE.md

## Objectif du projet

Entraîner des modèles de Machine Learning pour segmenter automatiquement un texte arabe en **unités narratives cohérentes** (khabar, pl. akhbar). Le texte cible est typiquement issu de sources historiographiques, littéraires ou journalistiques arabes.

## Structure du projet

```
khabar-segmentation/
├── CLAUDE.md                  # Ce fichier
├── .gitignore
├── requirements.txt
├── configs/                   # Hyperparamètres et configs de modèles
├── data/
│   ├── raw/                   # Textes arabes bruts (ne pas modifier)
│   ├── processed/             # Données après prétraitement
│   └── annotations/           # Annotations manuelles (format JSON/JSONL)
├── notebooks/                 # Exploration et visualisation (Jupyter)
├── scripts/
│   ├── preprocess.py          # Nettoyage et tokenisation
│   ├── train.py               # Boucle d'entraînement
│   └── evaluate.py            # Métriques d'évaluation
└── src/
    ├── __init__.py
    ├── data/                  # Loaders et datasets PyTorch/HuggingFace
    ├── models/                # Architectures de modèles
    └── utils/                 # Fonctions utilitaires (métriques, viz…)
```

## Conventions de code

- Python 3.10+
- Formatter : `black` (largeur 88)
- Linter : `ruff`
- Types : annotations obligatoires sur les fonctions publiques
- Langue des docstrings et commentaires : **français**
- Langue des commits git : **français**
- Nommage des fichiers de données : `{source}_{date}_{version}.{ext}`

## Rituel de synchronisation multi-appareils

À chaque changement d'environnement (VSCode → Claude Code web → smartphone), appliquer ce rituel :

```
[DÉBUT DE SESSION]  →  session-start.sh  (git pull --rebase)
       ... travail ...
[FIN DE SESSION]    →  session-end.sh    (git add -A + commit + push)
```

### Claude Code web (automatique)

- **Début** : le hook `SessionStart` exécute `session-start.sh` automatiquement.
- **Fin** : le hook `Stop` détecte les changements non poussés et me force à committer/pusher avant de s'arrêter.

### VSCode

- **Début** : la tâche `Début de session (pull)` se lance automatiquement à l'ouverture du dossier (si autorisé).
  - Ou manuellement : `Ctrl+Shift+P` → **Tasks: Run Task** → `Début de session (pull)`
- **Fin** : `Ctrl+Shift+P` → **Tasks: Run Task** → `Fin de session (commit + push)`

### Terminal (tous environnements)

```bash
# Début de session
bash scripts/session-start.sh

# Fin de session (message automatique)
bash scripts/session-end.sh

# Fin de session (message personnalisé)
bash scripts/session-end.sh "feat: ajout du tokeniseur arabe"
```

### Smartphone (Working Copy — iOS/Android)

1. Cloner le repo dans Working Copy
2. Activer **Auto-Fetch** dans les réglages du repo
3. Début : pull manuel depuis Working Copy
4. Fin : commit + push depuis Working Copy
5. Pour exécuter du code : ouvrir le notebook dans **Google Colab** via le bouton "Open in Colab"

---

## Workflow de développement

### Branching

- `main` — code stable, révisé
- `claude/...` — branches de travail automatiques (Claude Code)
- `feat/...` — nouvelles fonctionnalités
- `fix/...` — corrections de bugs
- `exp/...` — expériences ML (peut échouer, c'est OK)

Toujours travailler sur une branche dédiée, jamais directement sur `main`.

### Commandes courantes

```bash
# Installer les dépendances
pip install -r requirements.txt

# Formater le code
black src/ scripts/
ruff check src/ scripts/ --fix

# ===== SEGMENTATION BASELINE ISNAD-KHABAR =====
# Segmenter un texte arabe
python scripts/baseline_isnad_segmentation.py \
  --input "chemin/vers/texte.txt" \
  --target 613 \
  --samples 5

# Exemples sur corpus OpenITI
python scripts/baseline_isnad_segmentation.py \
  --input "/path/to/0406IbnHabibNaysaburi.CuqalaMajanin.JK010625-ara1" \
  --target 613

# Résultats dans results/segmentation_baseline.txt

# ===== PRÉTRAITEMENT ET ENTRAÎNEMENT (PHASE 2+) =====
# Prétraitement
python scripts/preprocess.py --input data/raw/ --output data/processed/

# Entraînement
python scripts/train.py --config configs/baseline.yaml

# Évaluation
python scripts/evaluate.py --model checkpoints/best/ --data data/processed/test/

# Lancer Jupyter
jupyter lab --no-browser --port 8888
```

### Gestion des données

- Les fichiers `data/raw/` ne sont **jamais modifiés** directement.
- Les données annotées sont en **JSONL** : un exemple par ligne.
- Format d'annotation minimal :
  ```json
  {"text": "...", "segments": [{"start": 0, "end": 42, "label": "khabar"}]}
  ```
- Les gros fichiers de données et checkpoints sont exclus du dépôt (`.gitignore`).
- Utiliser **DVC** ou un stockage externe (HuggingFace Hub, S3) pour versionner les données.

## Configuration multi-appareils

### VSCode (bureau/laptop)

Extensions recommandées :
- `ms-python.python` — support Python
- `ms-toolsai.jupyter` — notebooks inline
- `eamodio.gitlens` — historique git
- `ms-vscode-remote.remote-ssh` — connexion SSH si GPU distant

Paramètres `.vscode/settings.json` à créer localement :
```json
{
  "editor.formatOnSave": true,
  "python.formatting.provider": "black",
  "python.linting.ruffEnabled": true,
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

### Claude Code en ligne (claude.ai/code)

Claude Code peut lire, modifier et exécuter du code directement.
Opérations courantes déléguées à Claude :
- Générer du code de prétraitement pour de nouveaux formats de texte arabe
- Déboguer des boucles d'entraînement
- Analyser les métriques et proposer des ajustements d'hyperparamètres
- Rédiger des expériences dans `notebooks/`

### Smartphone (GitHub Mobile / Working Copy)

Pour les modifications légères depuis mobile :
- Consulter l'avancement via **GitHub Mobile**
- Éditer des fichiers de config YAML ou des notes dans `notebooks/`
- Valider des PRs et lire les résultats d'expériences
- Pour exécuter du code mobile : utiliser un notebook **Google Colab** lié au dépôt

## État actuel de la segmentation Isnad-Khabar

### ✅ Baseline v1 implémentée et testée (2026-04-14)

**Approche** : Détection linguistique basée sur la structure historique arabe
- Détecte les **isnads** (chaînes de transmission) via verbes spécifiques (حدثنا، أخبرنا، قال)
- Identifie les **transitions isnad→récit** (passage à 1ère personne, verbes narratifs)
- Segmente les **frontières de khabar** (ponctuation, nouveaux isnads)

**Résultats de test** :
| Texte | Akhbars | Écart | Couverture | Statut |
|-------|---------|-------|-----------|--------|
| 406IbnHabibNaysaburi (40.7K mots) | 708 | +15.5% vs 613 cible | 92.4% | ✅ Excellent |
| 392IbnIsmacilMisri (4.1K mots) | 39 | - | 90.8% | ✅ Cohérent |

**Scripts** :
- `scripts/baseline_isnad_segmentation.py` — **Utiliser celle-ci (v1)**
- `scripts/baseline_isnad_segmentation_v2.py` — Trop restrictive (37 akhbars)

**Documentation** :
- `BASELINE_RESULTS.md` — Analyse détaillée + recommandations
- `results/TEST_SUMMARY.md` — Résumé des tests sur deux textes
- `results/comparison_results.txt` — Analyse comparative

### Modèles et expériences à explorer

#### Phase 1 (Complétée)
1. ✅ **Baseline linguistique** — structure isnad→khabar [**FAIT**]
   - Écart 15.5% acceptable pour approche pure linguistique
   - Fondation robuste pour amélioration progressive

#### Phase 2 (Prochaine)
1. **Post-processing** (rapide, 1-2 jours)
   - Filtrer isnads < 5 mots → réduire écart de 15% → ~8%
   - Exclure citations coraniques
   - Résultat attendu : écart < 10%

2. **Fine-tune AraBERT / CAMeL-BERT** (1-2 semaines)
   - Annoter 100-200 exemples manuels
   - Token classifier (BIO tagging)
   - Résultat attendu : écart < 5%

3. **Modèles génératifs** (optionnel)
   - Jais, AraT5, ou autre LLM arabe pour zero-shot/few-shot
   - Combiner avec baseline pour robustesse

### Métriques d'évaluation

- **Pk** et **WindowDiff** — métriques standard de segmentation de texte
- **F1 segment-level** — précision/rappel sur les frontières de segments
- Calculées via la librairie `segeval`

## Variables d'environnement

Créer un fichier `.env` (non versionné) à la racine :
```bash
HF_TOKEN=...          # HuggingFace Hub token
WANDB_API_KEY=...     # Weights & Biases pour le suivi d'expériences
DATA_DIR=./data       # Chemin vers les données
CHECKPOINT_DIR=./checkpoints
```

Charger avec : `python-dotenv` ou `export $(cat .env | xargs)`

## Notes importantes pour Claude

- Les textes arabes sont en **UTF-8**, sens droite-à-gauche. Toujours vérifier l'encodage.
- Ne jamais modifier les fichiers dans `data/raw/`.
- Les checkpoints de modèles vont dans `checkpoints/` (ignoré par git).
- Avant tout entraînement, vérifier que les données sont bien équilibrées (ratio frontières/non-frontières).
- Préférer les configs YAML dans `configs/` aux arguments en dur dans les scripts.
- Documenter chaque expérience dans `notebooks/experiments_log.md`.

### Baseline Isnad-Segmentation (depuis 2026-04-14)

**État** : ✅ Phase 1 terminée (baseline v1)

**Références** :
- Script principal : `scripts/baseline_isnad_segmentation.py`
- Résumé de progression : `BASELINE_RESULTS.md` et `results/TEST_SUMMARY.md`
- Données de test : corpus OpenITI (406IbnHabib, 392IbnIsmacil)

**Résultat clé** : 708 akhbars détectés vs 613 cible = écart acceptable de 15.5%

### ✅ Phase 2 CAMeL-BERT — Résultats finaux (2026-04-21)

#### Contexte et erreur initiale d'analyse (2026-04-20)

Une première tentative de conversion token→char avait produit F1=19.6%, conduisant à la conclusion (erronée) que CAMeL-BERT ne détectait pas les bonnes frontières. L'erreur venait de la méthode de mapping : construire une séquence dédupliquée séquentielle depuis le fichier raw_inference créait un décalage de 8 477 tokens manquants (trous entre les chunks), faussant toutes les positions char.

#### Approche correcte : extraction directe via offsets globaux

Le fichier `results/camelbert_kitab_uqala_raw_inference.json` contient des **offsets globaux** (positions char dans le corpus original) pour chaque token. L'approche correcte est d'utiliser ces offsets directement, sans passer par un index séquentiel :

```python
# Pour chaque token dans raw_inference :
# - ignorer [PAD]/[CLS]/[SEP] et offsets [0,0]
# - si pred=1, enregistrer (char_start, probabilité, token_string)
# - dédupliquer par char_start en gardant la probabilité max

char_to_prob = {}
char_to_pred = {}
for tok, off, pred, prob in zip(tokens, offsets, preds, probs):
    if tok in SPECIAL or off == [0, 0]: continue
    cs = off[0]
    if cs not in char_to_prob or prob > char_to_prob[cs]:
        char_to_prob[cs] = prob
        char_to_pred[cs] = pred

# → 69 031 positions uniques, 16 165 avec pred=1
```

Puis clustering des boundary tokens contigus (gap ≤ 50 chars) → chaque cluster = un segment isnad. Le début de chaque cluster = frontière khabar candidate.

**Validation immédiate** : `أخبرنا` à char=746, prob=0.9797 → coïncide exactement avec la frontière gold #2.

#### Résultats — corpus kitab_uqala (vs 613 frontières gold)

| Tolérance | Précision | Rappel | F1 |
|-----------|-----------|--------|----|
| ±50 chars | 0.923 | 0.783 | **0.847** |
| ±80 chars | 0.935 | 0.793 | **0.858** |
| ±150 chars | 0.948 | 0.804 | **0.870** |
| **Baseline v4** (référence) | 0.873 | 0.820 | **0.846** |

**CAMeL-BERT dépasse la baseline v4** (F1=0.858 vs 0.846).

#### Précision de localisation des frontières détectées

Quand le modèle détecte une frontière, à quelle distance est-il du gold le plus proche ?

| Seuil | Pred dans ce seuil |
|-------|--------------------|
| ±10 chars | 86.5% (450/520) |
| ±20 chars | 89.8% (467/520) |
| ±80 chars | 93.5% (486/520) |
| ±500 chars | 99.8% (519/520) |

**Distance médiane : 3 chars** — la précision de localisation est quasi-parfaite. Il n'existe qu'1 seul vrai faux positif (>500 chars de tout gold).

#### Analyse des erreurs

**34 faux positifs (tol ±80)** : quasi tous sont de vrais isnads (`أخبرنا محمد قال :`) que l'annotateur gold a regroupés dans un khabar plus large. Le modèle est correct linguistiquement, c'est le niveau de granularité de l'annotation qui diffère.

**127 faux négatifs** : khabars sans formule d'isnad — passages philosophiques, introductions narratives, citations coraniques, débuts de sections. Le modèle ne peut structurellement pas les détecter (pas de signal isnad).

**Distribution des gaps entre clusters isnad** (= longueur des khabars détectés) :
- Médiane : 250 chars
- Max : 4 113 chars
- Gaps > 500 chars : 114 (khabars longs, souvent des récits en prose)

#### Scripts et fichiers de résultats

- `scripts/convert_boundary_tokens_direct.py` — **script principal, approche correcte**
- `results/camelbert_char_boundaries_v2.json` — 520 frontières khabar avec positions char
- `results/camelbert_boundary_tokens_clean.json` — boundary tokens bruts (depuis Colab)
- `results/camelbert_kitab_uqala_raw_inference.json` — inférence complète avec offsets globaux

#### Conclusion et prochaines étapes

CAMeL-BERT, utilisé via l'extraction directe d'offsets, **surpasse la baseline v4** et localise les frontières à 3 chars près (médiane). Sa limite principale est le rappel sur les khabars sans isnad (20% du corpus).

**Pistes d'amélioration du rappel** :
1. **Hybride** : CAMeL-BERT pour les khabars avec isnad + baseline v4 pour les khabars sans isnad (prose pure)
2. **Fine-tuning BIO** : ré-entraîner avec séquence BIO sur les vraies frontières khabar pour couvrir aussi les passages sans isnad
3. **Post-processing** : fusionner les clusters trop proches (< 100 chars) pour réduire les FP restants
