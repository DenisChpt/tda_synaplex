"""Experiment orchestrator — multi-run, multi-dataset, multi-architecture.

Handles:
  1. Training N runs with different seeds for statistical rigour
  2. TDA analysis with bootstrap confidence intervals
  3. Baseline overfitting detectors (early stopping, gradient/weight norms)
  4. Automatic change-point detection on topological metrics
  5. Serialisation of all results to disk
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from data.datasets import make_loaders
from experiment.trainer import Trainer, TrainHistory
from models.cnn_fashion import MODEL_REGISTRY
from tda.baselines import (
	detect_early_stopping,
	detect_gradient_explosion,
	detect_loss_gap,
	detect_weight_growth,
)
from tda.changepoint import (
	bootstrap_tda_metric,
	detect_changepoint_cusum,
	detect_changepoint_derivative,
)
from tda.topological_monitor import TopologicalMonitor

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
	"""Detection epochs for each method on a single run."""

	early_stopping_epoch: int | None = None
	gradient_explosion_epoch: int | None = None
	weight_growth_epoch: int | None = None
	loss_gap_epoch: int | None = None
	tda_cusum_pe1_epoch: int | None = None
	tda_derivative_pe1_epoch: int | None = None
	tda_cusum_betti1_epoch: int | None = None
	tda_derivative_betti1_epoch: int | None = None


@dataclass
class SingleRunResult:
	seed: int
	history: TrainHistory
	tda_monitor: TopologicalMonitor
	detection: DetectionResult
	bootstrap_stability: dict[int, dict] = field(default_factory=dict)
	"""Maps epoch → bootstrap results (mean, std per metric/dim)."""


@dataclass
class ExperimentResult:
	"""Aggregated results over multiple runs for one (dataset, model, subset_size) config."""

	dataset: str
	model_name: str
	subset_size: int | None
	n_epochs: int
	runs: list[SingleRunResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class Experiment:
	"""Run a complete experiment with multiple seeds and full analysis."""

	def __init__(self, config: dict[str, Any]) -> None:
		self.config = config
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.result: ExperimentResult | None = None

	@classmethod
	def from_config(cls, path: str | Path) -> Experiment:
		import yaml

		with open(path) as f:
			config = yaml.safe_load(f)
		return cls(config)

	def run(self) -> ExperimentResult:
		cfg = self.config
		dataset_name = cfg["dataset"]
		model_name = cfg["model"]
		subset_size = cfg.get("subset_size")
		n_epochs = cfg.get("n_epochs", 80)
		seeds = cfg.get("seeds", [42, 123, 456, 789, 1024])
		n_seeds = cfg.get("n_seeds", len(seeds))
		seeds = seeds[:n_seeds]

		tda_max_points = cfg.get("tda_max_points", 500)
		bootstrap_epochs = cfg.get("bootstrap_epochs", [1, 20, 40, 60, 80])
		n_bootstrap = cfg.get("n_bootstrap", 10)

		self.result = ExperimentResult(
			dataset=dataset_name,
			model_name=model_name,
			subset_size=subset_size,
			n_epochs=n_epochs,
		)

		for i, seed in enumerate(seeds):
			print(f"\n{'=' * 60}")
			print(
				f"  Run {i + 1}/{len(seeds)} | seed={seed} | {dataset_name} | {model_name} | n={subset_size}"
			)
			print(f"{'=' * 60}")

			run_result = self._single_run(
				dataset_name=dataset_name,
				model_name=model_name,
				subset_size=subset_size,
				n_epochs=n_epochs,
				seed=seed,
				tda_max_points=tda_max_points,
				bootstrap_epochs=bootstrap_epochs,
				n_bootstrap=n_bootstrap,
			)
			self.result.runs.append(run_result)

		return self.result

	def _single_run(
		self,
		dataset_name: str,
		model_name: str,
		subset_size: int | None,
		n_epochs: int,
		seed: int,
		tda_max_points: int,
		bootstrap_epochs: list[int],
		n_bootstrap: int,
	) -> SingleRunResult:
		# Reproducibility
		torch.manual_seed(seed)
		np.random.seed(seed)

		# Data
		data = make_loaders(
			dataset_name,
			batch_size=self.config.get("batch_size", 256),
			subset_size=subset_size,
			seed=seed,
		)

		# Model
		model_cls = MODEL_REGISTRY[model_name]
		model_kwargs: dict[str, Any] = {
			"n_classes": data["n_classes"],
			"bottleneck_dim": self.config.get("bottleneck_dim", 32),
			"dropout": self.config.get("dropout", 0.0),
		}
		if model_name == "mlp":
			model_kwargs["input_shape"] = data["input_shape"]
		else:
			model_kwargs["input_channels"] = data.get("input_channels", 1)

		model = model_cls(**model_kwargs)

		# Training
		optimizer = torch.optim.Adam(
			model.parameters(),
			lr=self.config.get("lr", 1e-3),
			weight_decay=self.config.get("weight_decay", 0.0),
		)
		trainer = Trainer(
			model=model,
			criterion=nn.CrossEntropyLoss(),
			optimizer=optimizer,
			device=self.device,
			activation_samples=tda_max_points,
		)
		history = trainer.fit(
			data["train_loader"],
			data["val_loader"],
			n_epochs=n_epochs,
			verbose=True,
		)

		# TDA analysis
		print("\n--- TDA Analysis ---")
		tda_monitor = TopologicalMonitor(
			max_dim=1, max_points=tda_max_points, seed=seed
		)
		for epoch, acts_dict in history.activations.items():
			layer_name = list(acts_dict.keys())[-1]
			activations = acts_dict[layer_name]
			result = tda_monitor.compute(activations, epoch=epoch)
			if epoch % 10 == 0 or epoch == 1:
				print(
					f"  Epoch {epoch:3d} | "
					f"b0={result.betti_numbers[0]:4d}  "
					f"b1={result.betti_numbers.get(1, 0):3d}  "
					f"PE0={result.persistent_entropy[0]:.3f}  "
					f"PE1={result.persistent_entropy.get(1, 0):.3f}"
				)

		# Bootstrap stability on selected epochs
		bootstrap_stability: dict[int, dict] = {}
		for ep in bootstrap_epochs:
			if ep in history.activations:
				layer_name = list(history.activations[ep].keys())[-1]
				acts = history.activations[ep][layer_name]
				bs = bootstrap_tda_metric(
					acts,
					TopologicalMonitor,
					n_bootstrap=n_bootstrap,
					max_points=tda_max_points,
					seed=seed,
				)
				bootstrap_stability[ep] = bs

		# Detection: baselines
		val_losses = [m.val_loss for m in history.metrics]
		train_losses = [m.train_loss for m in history.metrics]
		grad_norms = [m.gradient_norm for m in history.metrics]
		weight_norms = [m.weight_norm for m in history.metrics]

		det = DetectionResult(
			early_stopping_epoch=detect_early_stopping(val_losses, patience=10),
			gradient_explosion_epoch=detect_gradient_explosion(grad_norms),
			weight_growth_epoch=detect_weight_growth(weight_norms),
			loss_gap_epoch=detect_loss_gap(
				train_losses, val_losses, gap_threshold=0.1, sustained_epochs=5
			),
		)

		# Detection: TDA change-points
		_, pe1_vals = tda_monitor.get_metric_series("persistent_entropy", dim=1)
		_, betti1_vals = tda_monitor.get_metric_series("betti", dim=1)

		if pe1_vals:
			det.tda_cusum_pe1_epoch = detect_changepoint_cusum(
				pe1_vals, threshold=1.5, direction="up"
			)
			det.tda_derivative_pe1_epoch = detect_changepoint_derivative(
				pe1_vals, window=10, min_epoch=15
			)

		if betti1_vals:
			det.tda_cusum_betti1_epoch = detect_changepoint_cusum(
				betti1_vals, threshold=15.0, direction="up"
			)
			det.tda_derivative_betti1_epoch = detect_changepoint_derivative(
				betti1_vals, window=10, min_epoch=15
			)

		print(f"\n--- Detection results (seed={seed}) ---")
		print(f"  Early stopping (patience=10):  epoch {det.early_stopping_epoch}")
		print(f"  Gradient explosion:            epoch {det.gradient_explosion_epoch}")
		print(f"  Weight growth:                 epoch {det.weight_growth_epoch}")
		print(f"  Loss gap (>0.1, 5 epochs):     epoch {det.loss_gap_epoch}")
		print(f"  TDA CUSUM PE(H1):              epoch {det.tda_cusum_pe1_epoch}")
		print(f"  TDA derivative PE(H1):         epoch {det.tda_derivative_pe1_epoch}")
		print(f"  TDA CUSUM beta1:               epoch {det.tda_cusum_betti1_epoch}")
		print(
			f"  TDA derivative beta1:          epoch {det.tda_derivative_betti1_epoch}"
		)

		return SingleRunResult(
			seed=seed,
			history=history,
			tda_monitor=tda_monitor,
			detection=det,
			bootstrap_stability=bootstrap_stability,
		)

	def save(self, output_dir: str = "results") -> None:
		"""Save aggregated results."""
		if self.result is None:
			return
		out = Path(output_dir)
		out.mkdir(parents=True, exist_ok=True)

		tag = (
			f"{self.result.dataset}_{self.result.model_name}_n{self.result.subset_size}"
		)
		with open(out / f"{tag}_full_results.pkl", "wb") as f:
			pickle.dump(self.result, f, protocol=pickle.HIGHEST_PROTOCOL)

		# Save a lightweight JSON summary
		summary = self._build_summary()
		with open(out / f"{tag}_summary.json", "w") as f:
			json.dump(summary, f, indent=2, default=str)

		print(f"\nResults saved to {out}/{tag}_*")

	def _build_summary(self) -> dict:
		"""Build a JSON-serialisable summary of detection epochs across runs."""
		if self.result is None:
			return {}
		runs_det = []
		for r in self.result.runs:
			d = r.detection
			runs_det.append(
				{
					"seed": r.seed,
					"early_stopping": d.early_stopping_epoch,
					"gradient_explosion": d.gradient_explosion_epoch,
					"weight_growth": d.weight_growth_epoch,
					"loss_gap": d.loss_gap_epoch,
					"tda_cusum_pe1": d.tda_cusum_pe1_epoch,
					"tda_deriv_pe1": d.tda_derivative_pe1_epoch,
					"tda_cusum_betti1": d.tda_cusum_betti1_epoch,
					"tda_deriv_betti1": d.tda_derivative_betti1_epoch,
					"final_train_acc": r.history.metrics[-1].train_acc,
					"final_val_acc": r.history.metrics[-1].val_acc,
					"final_gap": r.history.metrics[-1].train_acc
					- r.history.metrics[-1].val_acc,
				}
			)

		# Aggregate stats
		def _stats(key: str) -> dict:
			vals = [r[key] for r in runs_det if r[key] is not None]
			if not vals:
				return {"mean": None, "std": None, "n_detected": 0}
			return {
				"mean": float(np.mean(vals)),
				"std": float(np.std(vals)),
				"n_detected": len(vals),
			}

		return {
			"config": {
				"dataset": self.result.dataset,
				"model": self.result.model_name,
				"subset_size": self.result.subset_size,
				"n_epochs": self.result.n_epochs,
				"n_runs": len(self.result.runs),
			},
			"per_run": runs_det,
			"aggregate": {
				"early_stopping": _stats("early_stopping"),
				"gradient_explosion": _stats("gradient_explosion"),
				"weight_growth": _stats("weight_growth"),
				"loss_gap": _stats("loss_gap"),
				"tda_cusum_pe1": _stats("tda_cusum_pe1"),
				"tda_deriv_pe1": _stats("tda_deriv_pe1"),
				"tda_cusum_betti1": _stats("tda_cusum_betti1"),
				"tda_deriv_betti1": _stats("tda_deriv_betti1"),
			},
		}
