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

## Modèles et expériences

### Approches à explorer (par ordre de priorité)

1. **Baseline CRF** — features linguistiques arabes (préfixes, ponctuation, longueur)
2. **Fine-tuning AraBERT / CAMeL-BERT** — séquence-à-séquence ou token classification
3. **Modèles génératifs** — Jais, AraT5, ou autre LLM arabe pour segmentation zero-shot/few-shot

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
