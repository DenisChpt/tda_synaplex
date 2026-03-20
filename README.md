# TDA-Synaplex

Detecting overfitting early in neural networks using persistent homology of internal activations, powered by [Synaplex](https://github.com/DenisChpt/synaplex) (Rust-based persistence engine).

This codebase supports the paper **"Early Detection of Overfitting via Persistent Homology Analysis of Deep Neural Network Activations"** (D. Chaput, February 2026).

## What it does

At each training epoch, the last-layer activations are extracted and their persistent homology (H0, H1) is computed. Topological invariants — persistent entropy PE(H1) and Betti number b1 — are tracked over time. A change-point detection on the smoothed derivative of PE(H1) signals the onset of overfitting, typically **well before** validation loss degrades.

On Fashion-MNIST/CNN, the TDA-based detector triggers around epoch 24, compared to epoch 77 for classical early stopping (patience=10), with a 100% detection rate across all configurations.

## Experiment design

8 configurations following a one-factor-at-a-time design, 5 seeds each:

- **Datasets**: Fashion-MNIST, CIFAR-10, MNIST
- **Architectures**: shallow CNN, deep CNN, MLP
- **Training set sizes**: 200, 500, 1000, 2000

Each run trains the model, computes TDA metrics per epoch, runs bootstrap stability analysis, and compares against four baseline detectors (early stopping, loss gap, gradient norm, weight norm).

## Project structure

```
configs/          YAML experiment configurations
src/
  main.py         Entry point (--config, --all, --list)
  experiment/     Training loop + orchestrator
  models/         CNN and MLP architectures
  data/           Dataset loaders (with subset sampling)
  tda/            Topological monitor, change-point detection, baselines
  visualization/  Plotting utilities
results/          Serialized results (pickle + JSON summaries)
article/          LaTeX source and figures for the paper
```

## Usage

```bash
# Install (requires Python >= 3.14)
# For GPU support, uncomment the appropriate PyTorch index in pyproject.toml first
# (ROCm for AMD, CUDA for NVIDIA)
uv sync

# Run a single experiment
python src/main.py --config configs/fashion_mnist_shallow_500.yaml

# Run all 8 configurations
python src/main.py --all

# List available configs
python src/main.py --list
```

## Dependencies

- PyTorch, torchvision
- [Synaplex](https://github.com/DenisChpt/synaplex) — persistent homology computation
- NumPy, PyYAML
