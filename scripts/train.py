"""Boucle d'entraînement principale."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main(config_path: Path) -> None:
    config = OmegaConf.load(config_path)
    print(f"Entraînement avec la config : {config_path}")
    print(OmegaConf.to_yaml(config))
    # TODO: implémenter la boucle d'entraînement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement du modèle de segmentation")
    parser.add_argument("--config", type=Path, required=True, help="Fichier de config YAML")
    args = parser.parse_args()
    main(args.config)
