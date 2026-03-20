"""Visualization module — persistence diagrams, loss curves, topological metrics.

All plot functions accept pre-computed data and return matplotlib Figure objects
so they can be saved or displayed interactively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray

from src.experiment.trainer import TrainHistory
from src.tda.topological_monitor import PersistenceResult, TopologicalMonitor

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


# ---------------------------------------------------------------------------
# 1. Persistence diagram
# ---------------------------------------------------------------------------


def plot_persistence_diagram(
    result: PersistenceResult,
    title: str = "Persistence Diagram",
) -> plt.Figure:
    """Plot a standard persistence diagram (birth vs death)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    labels = ["$H_0$", "$H_1$", "$H_2$"]

    all_finite_deaths: list[float] = []
    for dim, dgm in enumerate(result.diagrams):
        finite_mask = np.isfinite(dgm[:, 1])
        if finite_mask.any():
            all_finite_deaths.extend(dgm[finite_mask, 1].tolist())
        ax.scatter(
            dgm[finite_mask, 0],
            dgm[finite_mask, 1],
            s=15,
            alpha=0.6,
            color=colors[dim % len(colors)],
            label=labels[dim] if dim < len(labels) else f"$H_{dim}$",
        )

    max_val = max(all_finite_deaths) if all_finite_deaths else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, lw=1)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Loss curves (train vs val)
# ---------------------------------------------------------------------------


def plot_loss_curves(history: TrainHistory) -> plt.Figure:
    """Plot train and validation loss across epochs."""
    epochs = [m.epoch for m in history.metrics]
    train_loss = [m.train_loss for m in history.metrics]
    val_loss = [m.val_loss for m in history.metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Accuracy curves
# ---------------------------------------------------------------------------


def plot_accuracy_curves(history: TrainHistory) -> plt.Figure:
    """Plot train and validation accuracy across epochs."""
    epochs = [m.epoch for m in history.metrics]
    train_acc = [m.train_acc for m in history.metrics]
    val_acc = [m.val_acc for m in history.metrics]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_acc, label="Train Acc", linewidth=2)
    ax.plot(epochs, val_acc, label="Val Acc", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training & Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Topological metrics over epochs
# ---------------------------------------------------------------------------


def plot_topological_metrics(
    monitor: TopologicalMonitor,
    history: TrainHistory | None = None,
) -> plt.Figure:
    """Plot persistent entropy and Betti numbers alongside val loss.

    Creates a 2×2 grid:
        [0,0] PE(H_0) and PE(H_1) vs epoch
        [0,1] Betti numbers vs epoch
        [1,0] Total lifetimes vs epoch
        [1,1] Val loss overlay with PE(H_0) (dual axis)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # PE
    ax = axes[0, 0]
    for dim, label in [(0, "$PE(H_0)$"), (1, "$PE(H_1)$")]:
        epochs, vals = monitor.get_metric_series("persistent_entropy", dim=dim)
        if epochs:
            ax.plot(epochs, vals, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Persistent Entropy")
    ax.set_title("Persistent Entropy")
    ax.legend()

    # Betti
    ax = axes[0, 1]
    for dim, label in [(0, "$\\beta_0$"), (1, "$\\beta_1$")]:
        epochs, vals = monitor.get_metric_series("betti", dim=dim)
        if epochs:
            ax.plot(epochs, vals, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Betti Number")
    ax.set_title("Betti Numbers")
    ax.legend()

    # Total lifetimes
    ax = axes[1, 0]
    for dim, label in [(0, "$\\Sigma\\ell(H_0)$"), (1, "$\\Sigma\\ell(H_1)$")]:
        epochs, vals = monitor.get_metric_series("total_lifetime", dim=dim)
        if epochs:
            ax.plot(epochs, vals, label=label, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Lifetime")
    ax.set_title("Total Lifetimes")
    ax.legend()

    # Dual axis: val_loss + PE(H_0)
    ax = axes[1, 1]
    if history is not None:
        h_epochs = [m.epoch for m in history.metrics]
        val_loss = [m.val_loss for m in history.metrics]
        ax.plot(h_epochs, val_loss, "r-", label="Val Loss", linewidth=2)
        ax.set_ylabel("Val Loss", color="r")
        ax.tick_params(axis="y", labelcolor="r")

    ax2 = ax.twinx()
    epochs, pe_vals = monitor.get_metric_series("persistent_entropy", dim=0)
    if epochs:
        ax2.plot(epochs, pe_vals, "b--", label="$PE(H_0)$", linewidth=2)
        ax2.set_ylabel("$PE(H_0)$", color="b")
        ax2.tick_params(axis="y", labelcolor="b")
    ax.set_xlabel("Epoch")
    ax.set_title("Val Loss vs $PE(H_0)$")

    fig.suptitle("Topological Monitoring", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Swiss Roll 3D visualization
# ---------------------------------------------------------------------------


def plot_swiss_roll_3d(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    title: str = "Swiss Roll",
) -> plt.Figure:
    """3D scatter plot of the Swiss Roll coloured by class."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", s=5, alpha=0.7)
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="Class")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------


def save_fig(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save a figure and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
