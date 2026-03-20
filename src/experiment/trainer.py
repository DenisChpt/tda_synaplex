"""Generic training loop with activation extraction and baseline metric collection.

The Trainer handles the standard PyTorch train/eval cycle and, at the end of
each epoch, collects:
  - activations on a validation subsample (for TDA)
  - gradient norms (baseline: gradient monitoring)
  - weight norms (baseline: weight growth)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Protocol for models that expose activations
# ---------------------------------------------------------------------------


class ActivationModel(Protocol):
	def get_activations(self) -> OrderedDict[str, torch.Tensor]: ...
	def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
	def parameters(self): ...
	def train(self, mode: bool = True): ...
	def eval(self): ...


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------


@dataclass
class EpochMetrics:
	epoch: int
	train_loss: float
	val_loss: float
	train_acc: float
	val_acc: float
	gradient_norm: float = 0.0
	weight_norm: float = 0.0


@dataclass
class TrainHistory:
	metrics: list[EpochMetrics] = field(default_factory=list)
	activations: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
	"""Train a model and extract activations + baseline metrics each epoch."""

	def __init__(
		self,
		model: nn.Module,
		criterion: nn.Module,
		optimizer: torch.optim.Optimizer,
		device: str = "cpu",
		activation_samples: int = 1000,
	) -> None:
		self.model = model.to(device)
		self.criterion = criterion
		self.optimizer = optimizer
		self.device = device
		self.activation_samples = activation_samples
		self.history = TrainHistory()

	def fit(
		self,
		train_loader: DataLoader,
		val_loader: DataLoader,
		n_epochs: int = 50,
		verbose: bool = True,
	) -> TrainHistory:
		for epoch in range(1, n_epochs + 1):
			train_loss, train_acc, grad_norm = self._train_epoch(train_loader)
			val_loss, val_acc = self._eval_epoch(val_loader)
			w_norm = self._weight_norm()

			metrics = EpochMetrics(
				epoch=epoch,
				train_loss=train_loss,
				val_loss=val_loss,
				train_acc=train_acc,
				val_acc=val_acc,
				gradient_norm=grad_norm,
				weight_norm=w_norm,
			)
			self.history.metrics.append(metrics)

			acts = self._collect_activations(val_loader)
			self.history.activations[epoch] = acts

			if verbose:
				print(
					f"[Epoch {epoch:3d}] "
					f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
					f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  "
					f"grad_norm={grad_norm:.4f}  w_norm={w_norm:.2f}"
				)

		return self.history

	# ------------------------------------------------------------------
	# Private
	# ------------------------------------------------------------------

	def _train_epoch(self, loader: DataLoader) -> tuple[float, float, float]:
		self.model.train()
		total_loss = 0.0
		correct = 0
		total = 0
		batch_grad_norms: list[float] = []

		for X, y in loader:
			X, y = X.to(self.device), y.to(self.device)
			self.optimizer.zero_grad()
			logits = self.model(X)
			loss = self.criterion(logits, y)
			loss.backward()

			# Capture gradient norm before stepping
			gn = self._gradient_norm()
			batch_grad_norms.append(gn)

			self.optimizer.step()
			total_loss += loss.item() * X.size(0)
			correct += (logits.argmax(dim=1) == y).sum().item()
			total += X.size(0)

		avg_grad_norm = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0
		return total_loss / total, correct / total, avg_grad_norm

	@torch.no_grad()
	def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
		self.model.eval()
		total_loss = 0.0
		correct = 0
		total = 0
		for X, y in loader:
			X, y = X.to(self.device), y.to(self.device)
			logits = self.model(X)
			loss = self.criterion(logits, y)
			total_loss += loss.item() * X.size(0)
			correct += (logits.argmax(dim=1) == y).sum().item()
			total += X.size(0)
		return total_loss / total, correct / total

	def _gradient_norm(self) -> float:
		"""Compute the L2 norm of all gradients."""
		total = 0.0
		for p in self.model.parameters():
			if p.grad is not None:
				total += p.grad.data.norm(2).item() ** 2
		return total**0.5

	def _weight_norm(self) -> float:
		"""Compute the L2 norm of all parameters."""
		total = 0.0
		for p in self.model.parameters():
			total += p.data.norm(2).item() ** 2
		return total**0.5

	@torch.no_grad()
	def _collect_activations(self, loader: DataLoader) -> dict[str, np.ndarray]:
		self.model.eval()
		all_acts: dict[str, list[np.ndarray]] = {}
		collected = 0

		for X, _ in loader:
			X = X.to(self.device)
			self.model(X)
			batch_acts = self.model.get_activations()  # type: ignore[attr-defined]
			for name, tensor in batch_acts.items():
				arr = tensor.cpu().numpy()
				if arr.ndim > 2:
					arr = arr.reshape(arr.shape[0], -1)
				all_acts.setdefault(name, []).append(arr)
			collected += X.size(0)
			if collected >= self.activation_samples:
				break

		result: dict[str, np.ndarray] = {}
		for name, arrays in all_acts.items():
			merged = np.concatenate(arrays, axis=0)[: self.activation_samples]
			result[name] = merged
		return result
