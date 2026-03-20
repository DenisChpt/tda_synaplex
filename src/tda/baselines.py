"""Baseline overfitting detectors for comparison with TDA-based detection.

Each detector consumes the training history and returns the epoch at which
overfitting is detected (or None if not detected).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 1. Early stopping (patience on val_loss)
# ---------------------------------------------------------------------------


def detect_early_stopping(
	val_losses: list[float],
	patience: int = 10,
) -> int | None:
	"""Return the epoch at which early stopping would have triggered.

	Uses the standard rule: stop when val_loss has not improved for
	``patience`` consecutive epochs.
	"""
	best_loss = float("inf")
	wait = 0
	for epoch, loss in enumerate(val_losses, start=1):
		if loss < best_loss - 1e-6:
			best_loss = loss
			wait = 0
		else:
			wait += 1
			if wait >= patience:
				return epoch
	return None


# ---------------------------------------------------------------------------
# 2. Gradient norm monitoring
# ---------------------------------------------------------------------------


def detect_gradient_explosion(
	gradient_norms: list[float],
	window: int = 10,
	threshold_factor: float = 2.0,
) -> int | None:
	"""Detect the epoch where the gradient norm starts growing abnormally.

	Returns the first epoch where the gradient norm exceeds
	``threshold_factor * rolling_mean`` over a sliding window.
	"""
	if len(gradient_norms) < window + 1:
		return None
	norms = np.array(gradient_norms)
	for i in range(window, len(norms)):
		rolling_mean = norms[i - window : i].mean()
		if norms[i] > threshold_factor * rolling_mean:
			return i + 1  # 1-indexed epochs
	return None


# ---------------------------------------------------------------------------
# 3. Weight norm monitoring
# ---------------------------------------------------------------------------


def detect_weight_growth(
	weight_norms: list[float],
	window: int = 10,
	slope_threshold: float = 0.0,
) -> int | None:
	"""Detect the epoch where weight norm starts consistently growing.

	Fits a local linear regression over a sliding window. Returns the
	first epoch where the slope exceeds ``slope_threshold`` after an
	initial stabilisation period.
	"""
	if len(weight_norms) < window + 1:
		return None
	norms = np.array(weight_norms)
	x = np.arange(window, dtype=np.float64)
	x_centered = x - x.mean()
	for i in range(window, len(norms)):
		y = norms[i - window : i]
		slope = (x_centered * (y - y.mean())).sum() / (x_centered**2).sum()
		if slope > slope_threshold:
			return i + 1
	return None


# ---------------------------------------------------------------------------
# 4. Val-loss gap detector
# ---------------------------------------------------------------------------


def detect_loss_gap(
	train_losses: list[float],
	val_losses: list[float],
	gap_threshold: float = 0.1,
	sustained_epochs: int = 5,
) -> int | None:
	"""Detect the first epoch where the train-val gap exceeds a threshold
	for ``sustained_epochs`` consecutive epochs."""
	gaps = [v - t for t, v in zip(train_losses, val_losses)]
	count = 0
	for epoch, g in enumerate(gaps, start=1):
		if g > gap_threshold:
			count += 1
			if count >= sustained_epochs:
				return epoch - sustained_epochs + 1
		else:
			count = 0
	return None


# ---------------------------------------------------------------------------
# 5. Sharpness proxy (finite-difference Hessian trace approximation)
# ---------------------------------------------------------------------------


def compute_sharpness_hutchinson(
	model,
	criterion,
	data_loader,
	device: str = "cpu",
	n_samples: int = 5,
	epsilon: float = 1e-3,
) -> float:
	"""Approximate the trace of the Hessian using Hutchinson's estimator.

	This is a lightweight proxy for loss-landscape sharpness.
	Higher sharpness correlates with worse generalisation.

	Uses finite differences:  Tr(H) ≈ E_v[ v^T H v ]
	where H v ≈ (∇L(θ+εv) - ∇L(θ-εv)) / (2ε) and v is Rademacher.
	"""
	import torch

	model.eval()
	params = [p for p in model.parameters() if p.requires_grad]
	total_trace = 0.0

	for _ in range(n_samples):
		# Random Rademacher vector
		vs = [torch.randint_like(p, 0, 2).float() * 2 - 1 for p in params]

		# Perturb +ε
		with torch.no_grad():
			for p, v in zip(params, vs):
				p.add_(v, alpha=epsilon)
		loss_plus = _compute_loss(model, criterion, data_loader, device)

		# Perturb -2ε (from +ε to -ε)
		with torch.no_grad():
			for p, v in zip(params, vs):
				p.add_(v, alpha=-2 * epsilon)
		loss_minus = _compute_loss(model, criterion, data_loader, device)

		# Restore original
		with torch.no_grad():
			for p, v in zip(params, vs):
				p.add_(v, alpha=epsilon)

		# Hessian-vector product approximation: v^T H v ≈ (L+ - L-) / ε²
		# (second-order finite difference for quadratic form)
		n_params = sum(p.numel() for p in params)
		total_trace += (loss_plus - loss_minus) / (2 * epsilon**2)

	return total_trace / n_samples


def _compute_loss(model, criterion, data_loader, device: str) -> float:
	"""Compute average loss on the first batch of data_loader."""
	import torch

	model.eval()
	with torch.no_grad():
		for X, y in data_loader:
			X, y = X.to(device), y.to(device)
			logits = model(X)
			return criterion(logits, y).item()
	return 0.0
