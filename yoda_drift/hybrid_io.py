"""Small shared helpers for loading YODA checkpoints."""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def load_matching(module: nn.Module, weights: dict[str, Any]) -> int:
    current = module.state_dict()
    matching = {
        key: value
        for key, value in weights.items()
        if key in current and value.shape == current[key].shape
    }
    module.load_state_dict(matching, strict=False)
    return len(matching)
