"""CNN architectures for image classification with TDA-compatible bottlenecks.

All models expose ``get_activations()`` returning an OrderedDict with at least
a ``"bottleneck"`` key capturing the dense representation before the classifier.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helper mixin for bottleneck hook management
# ---------------------------------------------------------------------------


class _BottleneckHookMixin:
	"""Mixin that registers a forward hook on the first ReLU in self.bottleneck."""

	_activations: OrderedDict[str, torch.Tensor]
	_hooks: list
	bottleneck: nn.Sequential

	def _register_bottleneck_hook(self) -> None:
		for module in self.bottleneck:
			if isinstance(module, nn.ReLU):
				hook = module.register_forward_hook(self._bottleneck_hook)
				self._hooks.append(hook)

	def _bottleneck_hook(
		self, _module: nn.Module, _input: tuple, output: torch.Tensor
	) -> None:
		self._activations["bottleneck"] = output.detach()

	def get_activations(self) -> OrderedDict[str, torch.Tensor]:
		return OrderedDict(self._activations)

	def remove_hooks(self) -> None:
		for h in self._hooks:
			h.remove()
		self._hooks.clear()


# ---------------------------------------------------------------------------
# CNN Shallow (2 conv blocks) — original architecture
# ---------------------------------------------------------------------------


class CNNShallow(nn.Module, _BottleneckHookMixin):
	"""Lightweight 2-block CNN.

	Architecture: [Conv-BN-ReLU-Pool]×2 → AdaptiveAvgPool → Dense bottleneck → Classifier
	"""

	def __init__(
		self,
		input_channels: int = 1,
		n_classes: int = 10,
		bottleneck_dim: int = 32,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self._activations: OrderedDict[str, torch.Tensor] = OrderedDict()
		self._hooks: list = []

		self.conv_blocks = nn.Sequential(
			nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.bottleneck = nn.Sequential(
			nn.Linear(64, bottleneck_dim),
			nn.ReLU(),
		)
		if dropout > 0:
			self.bottleneck.append(nn.Dropout(dropout))
		self.classifier = nn.Linear(bottleneck_dim, n_classes)
		self._register_bottleneck_hook()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._activations.clear()
		h = self.conv_blocks(x)
		h = self.pool(h).flatten(1)
		h = self.bottleneck(h)
		return self.classifier(h)


# ---------------------------------------------------------------------------
# CNN Deep (4 conv blocks) — increased capacity
# ---------------------------------------------------------------------------


class CNNDeep(nn.Module, _BottleneckHookMixin):
	"""Deeper 4-block CNN with more capacity (prone to overfitting on small data).

	Architecture: [Conv-BN-ReLU-Pool]×4 → AdaptiveAvgPool → Dense bottleneck → Classifier
	"""

	def __init__(
		self,
		input_channels: int = 1,
		n_classes: int = 10,
		bottleneck_dim: int = 32,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self._activations: OrderedDict[str, torch.Tensor] = OrderedDict()
		self._hooks: list = []

		self.conv_blocks = nn.Sequential(
			# Block 1
			nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# Block 2
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2),
			# Block 3
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			# Block 4
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
		)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.bottleneck = nn.Sequential(
			nn.Linear(128, bottleneck_dim),
			nn.ReLU(),
		)
		if dropout > 0:
			self.bottleneck.append(nn.Dropout(dropout))
		self.classifier = nn.Linear(bottleneck_dim, n_classes)
		self._register_bottleneck_hook()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._activations.clear()
		h = self.conv_blocks(x)
		h = self.pool(h).flatten(1)
		h = self.bottleneck(h)
		return self.classifier(h)


# ---------------------------------------------------------------------------
# MLP (no convolutions) — for comparison
# ---------------------------------------------------------------------------


class MLPClassifier(nn.Module, _BottleneckHookMixin):
	"""Fully-connected MLP that flattens the input image.

	Architecture: Flatten → [Linear-ReLU]×2 → Dense bottleneck → Classifier
	"""

	def __init__(
		self,
		input_shape: tuple[int, ...] = (1, 28, 28),
		n_classes: int = 10,
		bottleneck_dim: int = 32,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self._activations: OrderedDict[str, torch.Tensor] = OrderedDict()
		self._hooks: list = []

		flat_dim = 1
		for d in input_shape:
			flat_dim *= d

		self.features = nn.Sequential(
			nn.Flatten(),
			nn.Linear(flat_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
		)
		self.bottleneck = nn.Sequential(
			nn.Linear(128, bottleneck_dim),
			nn.ReLU(),
		)
		if dropout > 0:
			self.bottleneck.append(nn.Dropout(dropout))
		self.classifier = nn.Linear(bottleneck_dim, n_classes)
		self._register_bottleneck_hook()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._activations.clear()
		h = self.features(x)
		h = self.bottleneck(h)
		return self.classifier(h)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
	"cnn_shallow": CNNShallow,
	"cnn_deep": CNNDeep,
	"mlp": MLPClassifier,
}
