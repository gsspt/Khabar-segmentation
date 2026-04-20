"""Prétraitement des textes arabes bruts en données structurées pour l'entraînement."""

import argparse
import json
from pathlib import Path


def main(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: implémenter le pipeline de prétraitement
    print(f"Prétraitement de {input_dir} → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement des données arabes")
    parser.add_argument("--input", type=Path, required=True, help="Dossier des données brutes")
    parser.add_argument("--output", type=Path, required=True, help="Dossier de sortie")
    args = parser.parse_args()
    main(args.input, args.output)
