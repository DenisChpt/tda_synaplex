"""Topological Monitor — Persistent Homology analysis of neural activations using synaplex.

This module provides the core TDA functionality: given a tensor of activations
(shape [N, D]), it computes persistence diagrams, Betti numbers, lifetime
statistics, and persistent entropy using synaplex (Rust-based engine).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import synaplex

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PersistenceResult:
	"""Stores all topological summaries for a single activation snapshot."""

	diagrams: list[NDArray[np.float64]]  # one array per homology dim
	betti_numbers: dict[int, int]  # {0: b0, 1: b1, ...}
	total_lifetimes: dict[int, float]  # sum of lifetimes per dim
	persistent_entropy: dict[int, float]  # PE per dim
	mean_lifetime: dict[int, float]  # mean lifetime per dim
	max_lifetime: dict[int, float]  # max lifetime per dim
	epoch: int | None = None
	metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------


class TopologicalMonitor:
	"""Compute and track persistent homology of activation tensors.

	Parameters
	----------
	max_dim : int
		Maximum homology dimension to compute (default 1 → H_0 and H_1).
	max_points : int
		If the activation tensor has more rows than this, a random subsample
		is drawn to keep computation tractable.
	seed : int
		Random seed for reproducible subsampling.
	modulus : int
		Prime modulus for coefficient field (default 2 for Z/2Z).
	"""

	def __init__(
		self,
		max_dim: int = 1,
		max_points: int = 1000,
		seed: int = 42,
		modulus: int = 2,
	) -> None:
		self.max_dim = max_dim
		self.max_points = max_points
		self.rng = np.random.default_rng(seed)
		self.modulus = modulus
		self.history: list[PersistenceResult] = []

	def compute(
		self,
		activations: NDArray[np.float64],
		epoch: int | None = None,
		store: bool = True,
	) -> PersistenceResult:
		"""Run the full persistent homology pipeline on *activations*."""
		X = self._subsample(activations)
		diagrams = self._compute_diagrams(X)

		betti: dict[int, int] = {}
		total_lt: dict[int, float] = {}
		pe: dict[int, float] = {}
		mean_lt: dict[int, float] = {}
		max_lt: dict[int, float] = {}

		for dim, dgm in enumerate(diagrams):
			lifetimes = self._lifetimes(dgm)
			betti[dim] = self._betti_number(dgm)
			total_lt[dim] = float(lifetimes.sum()) if len(lifetimes) > 0 else 0.0
			pe[dim] = self._persistent_entropy(lifetimes)
			mean_lt[dim] = float(lifetimes.mean()) if len(lifetimes) > 0 else 0.0
			max_lt[dim] = float(lifetimes.max()) if len(lifetimes) > 0 else 0.0

		result = PersistenceResult(
			diagrams=diagrams,
			betti_numbers=betti,
			total_lifetimes=total_lt,
			persistent_entropy=pe,
			mean_lifetime=mean_lt,
			max_lifetime=max_lt,
			epoch=epoch,
		)
		if store:
			self.history.append(result)
		return result

	def save_history(self, path: str | Path) -> None:
		"""Persist the full history list as a pickle."""
		path = Path(path)
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, "wb") as f:
			pickle.dump(self.history, f, protocol=pickle.HIGHEST_PROTOCOL)

	def load_history(self, path: str | Path) -> None:
		"""Load a previously saved history."""
		with open(path, "rb") as f:
			self.history = pickle.load(f)  # noqa: S301

	def get_metric_series(
		self, metric: str, dim: int = 0
	) -> tuple[list[int], list[float]]:
		"""Extract a time-series of a given metric across epochs."""
		accessor = {
			"betti": "betti_numbers",
			"total_lifetime": "total_lifetimes",
			"persistent_entropy": "persistent_entropy",
			"mean_lifetime": "mean_lifetime",
			"max_lifetime": "max_lifetime",
		}
		if metric not in accessor:
			raise ValueError(
				f"Unknown metric '{metric}'. Choose from {list(accessor.keys())}."
			)
		attr = accessor[metric]
		epochs: list[int] = []
		values: list[float] = []
		for r in self.history:
			if r.epoch is not None:
				epochs.append(r.epoch)
				values.append(float(getattr(r, attr)[dim]))
		return epochs, values

	def _subsample(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Randomly subsample to at most ``max_points`` rows."""
		n = X.shape[0]
		if n <= self.max_points:
			return X
		idx = self.rng.choice(n, size=self.max_points, replace=False)
		return X[idx]

	def _compute_diagrams(self, X: NDArray[np.float64] | Tensor) -> list[NDArray[np.float64]]:
		"""Run synaplex on the point cloud and return persistence diagrams.

		Returns list of diagrams, one per dimension (H_0, H_1, ..., H_max_dim).
		"""
		# Ensure X is a contiguous float64 array
		if isinstance(X, Tensor):
			X = X.numpy()
		X = np.ascontiguousarray(X, dtype=np.float64)

		# synaplex returns: [[dim, birth, death], ...]
		result = synaplex.persistence_diagram(
			X,
			max_dim=self.max_dim,
			modulus=self.modulus,
		)

		# Convert to list of diagrams per dimension (like ripser format)
		diagrams = []
		for dim in range(self.max_dim + 1):
			# Filter pairs for this dimension
			mask = result[:, 0] == dim
			dim_pairs = result[mask]

			# Convert to (birth, death) format
			if len(dim_pairs) > 0:
				dgm = dim_pairs[:, 1:3]  # birth, death columns
			else:
				dgm = np.zeros((0, 2), dtype=np.float64)
			diagrams.append(dgm)

		return diagrams

	@staticmethod
	def _lifetimes(diagram: NDArray[np.float64]) -> NDArray[np.float64]:
		"""Compute finite lifetimes from a persistence diagram."""
		finite_mask = np.isfinite(diagram[:, 1])
		dgm = diagram[finite_mask]
		if len(dgm) == 0:
			return np.array([], dtype=np.float64)
		return dgm[:, 1] - dgm[:, 0]

	@staticmethod
	def _betti_number(diagram: NDArray[np.float64]) -> int:
		"""Count features with *finite* lifetime."""
		finite_mask = np.isfinite(diagram[:, 1])
		return int(finite_mask.sum())

	@staticmethod
	def _persistent_entropy(lifetimes: NDArray[np.float64]) -> float:
		"""Compute the persistent entropy of a set of lifetimes."""
		if len(lifetimes) == 0 or lifetimes.sum() == 0:
			return 0.0
		L = lifetimes.sum()
		p = lifetimes / L
		p = p[p > 0]
		return float(-np.sum(p * np.log(p)))
