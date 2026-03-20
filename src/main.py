"""Entry point for TDA overfitting detection experiments using synaplex.

Usage
-----
    # Single experiment
    python main.py --config configs/fashion_mnist_shallow_500.yaml

    # Run all experiments sequentially
    python main.py --all

    # List available configs
    python main.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment.experiment import Experiment

ALL_CONFIGS = [
    # Multi-dataset (same architecture, same subset)
    "configs/fashion_mnist_shallow_500.yaml",
    "configs/cifar10_shallow_500.yaml",
    "configs/mnist_shallow_500.yaml",
    # Multi-architecture (same dataset, same subset)
    "configs/fashion_mnist_deep_500.yaml",
    "configs/fashion_mnist_mlp_500.yaml",
    # Multi-subset (same dataset, same architecture)
    "configs/fashion_mnist_shallow_200.yaml",
    "configs/fashion_mnist_shallow_1000.yaml",
    "configs/fashion_mnist_shallow_2000.yaml",
]


def run_single(config_path: str) -> None:
    print(f"\n{'#' * 70}")
    print(f"# Running: {config_path}")
    print(f"{'#' * 70}")
    exp = Experiment.from_config(config_path)
    exp.run()
    exp.save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TDA Overfitting Detection with synaplex — Full Experiment Suite"
    )
    parser.add_argument("--config", type=str, help="Path to a single YAML config file")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--list", action="store_true", help="List available configs")
    args = parser.parse_args()

    if args.list:
        print("Available experiment configs:")
        for c in ALL_CONFIGS:
            exists = "OK" if Path(c).exists() else "MISSING"
            print(f"  [{exists}] {c}")
        return

    if args.all:
        for config_path in ALL_CONFIGS:
            if Path(config_path).exists():
                run_single(config_path)
            else:
                print(f"SKIPPING (not found): {config_path}")
        return

    if args.config:
        run_single(args.config)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
