"""Évaluation d'un modèle entraîné sur les données de test."""

import argparse
from pathlib import Path


def main(model_dir: Path, data_dir: Path) -> None:
    print(f"Évaluation de {model_dir} sur {data_dir}")
    # TODO: implémenter l'évaluation (Pk, WindowDiff, F1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation du modèle")
    parser.add_argument("--model", type=Path, required=True, help="Dossier du checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Dossier des données de test")
    args = parser.parse_args()
    main(args.model, args.data)
