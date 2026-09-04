"""Inspectable constraint scorers with safe, schema-validated checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

from warmcg.dataset import LabelTarget
from warmcg.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

CHECKPOINT_SCHEMA_VERSION = "warmcg-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class ConstraintScorerConfig:
    hidden_dim: int = 64
    hidden_layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("model dimensions must be positive")


def _mlp(input_dim: int, hidden_dim: int, hidden_layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    width = input_dim
    for _ in range(hidden_layers):
        modules.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = hidden_dim
    modules.append(nn.Linear(width, 1))
    return nn.Sequential(*modules)


class ConstraintScorer(nn.Module):
    """MLP ranking candidate SECs from solver-free, complement-invariant features."""

    network: nn.Sequential
    feature_mean: Tensor
    feature_scale: Tensor

    def __init__(self, config: ConstraintScorerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ConstraintScorerConfig()
        self.network = _mlp(len(FEATURE_NAMES), self.config.hidden_dim, self.config.hidden_layers)
        self.register_buffer("feature_mean", torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer("feature_scale", torch.ones(len(FEATURE_NAMES)))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_normalization(self, mean: np.ndarray, scale: np.ndarray) -> None:
        if mean.shape != (len(FEATURE_NAMES),) or scale.shape != (len(FEATURE_NAMES),):
            raise ValueError("normalization vectors have incompatible dimensions")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("normalization vectors must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("normalization scales must be positive")
        with torch.no_grad():
            self.feature_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
            self.feature_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"constraint features must have shape [candidates, {len(FEATURE_NAMES)}]"
            )
        normalized = (features - self.feature_mean) / self.feature_scale
        logits = cast(Tensor, self.network(normalized)).squeeze(-1)
        if not torch.all(torch.isfinite(logits)):
            raise RuntimeError("constraint scorer produced non-finite logits")
        return logits


def score_feature_matrix(
    model: ConstraintScorer,
    features: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Score an arbitrary candidate matrix without retaining autograd state."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("feature matrix has an incompatible shape")
    if not np.all(np.isfinite(features)):
        raise ValueError("feature matrix contains non-finite values")
    rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            tensor = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=model.device,
            )
            rows.append(model(tensor).detach().cpu().double().numpy())
    result = np.concatenate(rows) if rows else np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("constraint scoring returned non-finite values")
    return result


def _checkpoint_header(
    model: ConstraintScorer,
    *,
    target: LabelTarget,
    metadata: dict[str, object],
) -> dict[str, str]:
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": json.dumps(FEATURE_NAMES),
        "model_type": "constraint_scorer",
        "label_target": target,
        "model_config": json.dumps(asdict(model.config), sort_keys=True),
        "metadata": json.dumps(metadata, sort_keys=True),
    }


def save_checkpoint(
    model: ConstraintScorer,
    path: str | Path,
    *,
    target: LabelTarget,
    metadata: dict[str, object] | None = None,
) -> None:
    """Save weights and normalization buffers without pickle deserialization."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(
        tensors,
        str(output),
        metadata=_checkpoint_header(model, target=target, metadata=metadata or {}),
    )


def _config_integer(config: dict[str, object], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint model field {name!r} must be an integer")
    return value


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ConstraintScorer, LabelTarget, dict[str, object]]:
    """Load a schema-compatible constraint scorer."""

    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        header = handle.metadata()
        tensors = {
            key: handle.get_tensor(key)
            for key in handle.keys()  # noqa: SIM118 -- Safetensors is not iterable.
        }
    if header is None:
        raise ValueError("checkpoint metadata is missing")
    if header.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if header.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("checkpoint feature schema is incompatible")
    if header.get("model_type") != "constraint_scorer":
        raise ValueError("checkpoint model type is unsupported")
    if tuple(json.loads(header.get("feature_names", "[]"))) != FEATURE_NAMES:
        raise ValueError("checkpoint feature ordering is incompatible")
    target_raw = header.get("label_target")
    if target_raw not in {"invariant", "trajectory", "binding", "initial"}:
        raise ValueError("checkpoint label target is unsupported")
    target = cast(LabelTarget, target_raw)
    raw_config: object = json.loads(header.get("model_config", "{}"))
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint model configuration is invalid")
    config = cast(dict[str, object], raw_config)
    model = ConstraintScorer(
        ConstraintScorerConfig(
            hidden_dim=_config_integer(config, "hidden_dim"),
            hidden_layers=_config_integer(config, "hidden_layers"),
        )
    )
    expected_keys = set(model.state_dict())
    if set(tensors) != expected_keys:
        raise ValueError("checkpoint tensor keys are incompatible")
    model.load_state_dict(tensors, strict=True)
    model.to(device)
    raw_metadata: object = json.loads(header.get("metadata", "{}"))
    if not isinstance(raw_metadata, dict):
        raise ValueError("checkpoint metadata payload is invalid")
    return model, target, cast(dict[str, object], raw_metadata)
