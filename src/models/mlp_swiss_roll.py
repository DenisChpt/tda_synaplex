"""MLP for the Swiss Roll unfolding task (Phase 1).

Architecture: 3 → 64 → 32 → 16 → n_classes

The intermediate layers are deliberately narrow so we can inspect
*every* hidden representation in the persistence analysis.  Hooks are
registered to capture activations during a forward pass.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class SwissRollMLP(nn.Module):
	"""Simple MLP with hooks for extracting layer-wise activations.

	Parameters
	----------
	input_dim : int
		Dimension of input features (3 for Swiss Roll).
	n_classes : int
		Number of output classes.
	hidden_dims : tuple[int, ...]
		Widths of hidden layers.
	dropout : float
		Dropout probability (set > 0 to delay overfitting for comparison).
	"""

	def __init__(
		self,
		input_dim: int = 3,
		n_classes: int = 6,
		hidden_dims: tuple[int, ...] = (64, 32, 16),
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.layer_names: list[str] = []
		self._activations: OrderedDict[str, torch.Tensor] = OrderedDict()

		layers: list[nn.Module] = []
		prev = input_dim
		for i, h in enumerate(hidden_dims):
			name = f"hidden_{i}"
			self.layer_names.append(name)
			layers.append(nn.Linear(prev, h))
			layers.append(nn.ReLU())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			prev = h

		self.features = nn.Sequential(*layers)
		self.classifier = nn.Linear(prev, n_classes)

		# Register forward hooks on every ReLU
		self._hooks: list[torch.utils.hooks.RemovableHook] = []
		self._register_hooks()

	def _register_hooks(self) -> None:
		relu_idx = 0
		for module in self.features:
			if isinstance(module, nn.ReLU):
				name = self.layer_names[relu_idx]
				hook = module.register_forward_hook(self._make_hook(name))
				self._hooks.append(hook)
				relu_idx += 1

	def _make_hook(self, name: str):
		def hook_fn(_module: nn.Module, _input: tuple, output: torch.Tensor) -> None:
			self._activations[name] = output.detach()

		return hook_fn

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		self._activations.clear()
		h = self.features(x)
		return self.classifier(h)

	def get_activations(self) -> OrderedDict[str, torch.Tensor]:
		"""Return the activations captured during the last forward pass."""
		return OrderedDict(self._activations)

	def remove_hooks(self) -> None:
		for h in self._hooks:
			h.remove()
		self._hooks.clear()
