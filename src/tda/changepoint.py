"""Change-point detection for topological metric time-series.

Provides CUSUM-based and derivative-based methods to automatically detect
the epoch at which a topological metric transitions from one regime to
another (e.g. compression → fragmentation in PE(H₁)).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def detect_changepoint_cusum(
	values: list[float] | NDArray[np.float64],
	threshold: float = 1.0,
	drift: float = 0.0,
	direction: str = "up",
) -> int | None:
	"""Detect a change-point using the CUSUM algorithm.

	Parameters
	----------
	values : array-like
		Time-series of metric values (one per epoch).
	threshold : float
		CUSUM alarm threshold (h). Larger = less sensitive.
	drift : float
		Allowance parameter (k). Subtracted from each deviation.
	direction : str
		``"up"`` to detect an increase, ``"down"`` for a decrease.

	Returns
	-------
	int or None
		1-indexed epoch of the detected change-point, or None.
	"""
	arr = np.asarray(values, dtype=np.float64)
	if len(arr) < 3:
		return None
	mu = arr[: max(3, len(arr) // 4)].mean()  # baseline from first quarter
	s = 0.0
	for i, x in enumerate(arr):
		if direction == "up":
			s = max(0.0, s + (x - mu) - drift)
		else:
			s = max(0.0, s + (mu - x) - drift)
		if s > threshold:
			return i + 1  # 1-indexed
	return None


def detect_changepoint_derivative(
	values: list[float] | NDArray[np.float64],
	window: int = 10,
	min_epoch: int = 15,
) -> int | None:
	"""Detect the epoch where the smoothed derivative changes sign.

	Specifically, find the first epoch (after ``min_epoch``) where the
	smoothed first derivative transitions from negative/zero to positive,
	indicating the start of an increasing trend (e.g. PE starting to rise).

	Parameters
	----------
	values : array-like
		Time-series of metric values.
	window : int
		Smoothing window for the moving average.
	min_epoch : int
		Ignore the first epochs (transient regime).

	Returns
	-------
	int or None
		1-indexed epoch of the detected transition, or None.
	"""
	arr = np.asarray(values, dtype=np.float64)
	if len(arr) < min_epoch + window:
		return None

	# Smooth with moving average
	kernel = np.ones(window) / window
	smoothed = np.convolve(arr, kernel, mode="valid")

	# First derivative of smoothed signal
	deriv = np.diff(smoothed)

	# Offset: convolve 'valid' mode starts at index (window-1)
	# diff shifts by 1 more, so deriv[i] corresponds to epoch (window-1) + i + 1
	offset = window  # 1-indexed epoch of deriv[0]

	# Find first sign change from ≤0 to >0 after min_epoch
	for i in range(max(0, min_epoch - offset), len(deriv) - 1):
		if deriv[i] <= 0 and deriv[i + 1] > 0:
			return offset + i + 1  # 1-indexed epoch
	return None


def bootstrap_tda_metric(
	activations: NDArray[np.float64],
	monitor_cls,
	n_bootstrap: int = 10,
	max_points: int = 500,
	seed: int = 42,
) -> dict[str, dict[int, tuple[float, float]]]:
	"""Compute bootstrap confidence intervals for TDA metrics.

	Resamples ``max_points`` from ``activations`` ``n_bootstrap`` times,
	computes persistence, and returns mean ± std for each metric/dim.

	Returns
	-------
	dict mapping metric_name → {dim: (mean, std)}
	"""
	rng = np.random.default_rng(seed)
	results: dict[str, list[dict[int, float]]] = {
		"betti": [],
		"persistent_entropy": [],
		"total_lifetime": [],
	}
	# Use 80% subsample (without replacement) for variability
	subsample_size = min(int(activations.shape[0] * 0.8), max_points)
	for i in range(n_bootstrap):
		idx = rng.choice(activations.shape[0], size=subsample_size, replace=False)
		subset = activations[idx]
		mon = monitor_cls(max_dim=1, max_points=max_points, seed=seed + i)
		r = mon.compute(subset, store=False)
		results["betti"].append(r.betti_numbers)
		results["persistent_entropy"].append(r.persistent_entropy)
		results["total_lifetime"].append(r.total_lifetimes)

	summary: dict[str, dict[int, tuple[float, float]]] = {}
	for metric, runs in results.items():
		summary[metric] = {}
		for dim in [0, 1]:
			vals = [r[dim] for r in runs]
			summary[metric][dim] = (float(np.mean(vals)), float(np.std(vals)))
	return summary
