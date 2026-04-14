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

**Prochaines étapes** :
1. Post-processing pour réduire écart à ~8% (très rapide)
2. Fine-tune CAMeL-BERT sur 100-200 exemples annotés (1-2 semaines)
