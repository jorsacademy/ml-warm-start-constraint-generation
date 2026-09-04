"""Frozen train-once, evaluate-many constraint warm-start protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from warmcg.dataset import WarmStartDataset, generate_dataset
from warmcg.evaluation import EvaluationReport, evaluate_dataset
from warmcg.model import ConstraintScorerConfig, save_checkpoint
from warmcg.training import TrainingConfig, TrainingReport, train_constraint_scorer
from warmcg.utils import as_object_dict, finite_float, integer, sha256_json, string

RESEARCH_CONFIG_SCHEMA_VERSION = "warmcg-research-v1"


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    count: int
    node_count: int
    regime: str
    seed: int

    def __post_init__(self) -> None:
        if not self.name or self.count <= 0 or not 5 <= self.node_count <= 14:
            raise ValueError("scenario name, count, or node_count is invalid")
        if not self.regime or self.seed < 0:
            raise ValueError("scenario regime or seed is invalid")


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    training_count: int
    training_node_counts: tuple[int, ...]
    training_regimes: tuple[str, ...]
    training_seed: int
    validation_count: int
    validation_node_counts: tuple[int, ...]
    validation_regimes: tuple[str, ...]
    validation_seed: int
    hidden_dim: int
    hidden_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    negative_ratio: int
    minimum_negatives_per_instance: int
    patience: int
    warm_start_budget: int
    bootstrap_draws: int
    bootstrap_seed: int
    model_seed: int
    scenarios: tuple[ScenarioConfig, ...]

    def __post_init__(self) -> None:
        if self.training_count <= 0 or self.validation_count <= 0:
            raise ValueError("training and validation counts must be positive")
        if not self.training_node_counts or not self.validation_node_counts:
            raise ValueError("training and validation node counts must be nonempty")
        if not self.training_regimes or not self.validation_regimes:
            raise ValueError("training and validation regimes must be nonempty")
        if any(not 5 <= value <= 14 for value in self.training_node_counts):
            raise ValueError("training node count is outside the exact-labeling range")
        if any(not 5 <= value <= 14 for value in self.validation_node_counts):
            raise ValueError("validation node count is outside the exact-labeling range")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0.0:
            raise ValueError("training hyperparameters are invalid")
        if self.weight_decay < 0.0 or self.negative_ratio <= 0:
            raise ValueError("regularization or sampling settings are invalid")
        if self.minimum_negatives_per_instance <= 0 or self.patience <= 0:
            raise ValueError("sampling and early-stopping settings are invalid")
        if self.warm_start_budget < 0 or self.bootstrap_draws <= 0:
            raise ValueError("evaluation budgets are invalid")
        if not self.scenarios:
            raise ValueError("research configuration must define evaluation scenarios")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["schema_version"] = RESEARCH_CONFIG_SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ResearchConfig:
        if payload.get("schema_version") != RESEARCH_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported research configuration schema")
        raw_scenarios = payload.get("scenarios")
        if not isinstance(raw_scenarios, (list, tuple)):
            raise ValueError("research scenarios must be a JSON array")
        scenarios: list[ScenarioConfig] = []
        for entry in raw_scenarios:
            raw = as_object_dict(entry, name="research scenario")
            scenarios.append(
                ScenarioConfig(
                    name=string(raw.get("name"), name="scenario.name"),
                    count=integer(raw.get("count"), name="scenario.count", minimum=1),
                    node_count=integer(
                        raw.get("node_count"), name="scenario.node_count", minimum=5
                    ),
                    regime=string(raw.get("regime"), name="scenario.regime"),
                    seed=integer(raw.get("seed"), name="scenario.seed", minimum=0),
                )
            )
        return cls(
            training_count=integer(
                payload.get("training_count"), name="training_count", minimum=1
            ),
            training_node_counts=tuple(
                _integer_list(payload, "training_node_counts")
            ),
            training_regimes=tuple(_string_list(payload, "training_regimes")),
            training_seed=integer(
                payload.get("training_seed"), name="training_seed", minimum=0
            ),
            validation_count=integer(
                payload.get("validation_count"), name="validation_count", minimum=1
            ),
            validation_node_counts=tuple(
                _integer_list(payload, "validation_node_counts")
            ),
            validation_regimes=tuple(_string_list(payload, "validation_regimes")),
            validation_seed=integer(
                payload.get("validation_seed"), name="validation_seed", minimum=0
            ),
            hidden_dim=integer(payload.get("hidden_dim"), name="hidden_dim", minimum=1),
            hidden_layers=integer(
                payload.get("hidden_layers"), name="hidden_layers", minimum=1
            ),
            epochs=integer(payload.get("epochs"), name="epochs", minimum=1),
            batch_size=integer(payload.get("batch_size"), name="batch_size", minimum=1),
            learning_rate=finite_float(
                payload.get("learning_rate"), name="learning_rate"
            ),
            weight_decay=finite_float(payload.get("weight_decay"), name="weight_decay"),
            negative_ratio=integer(
                payload.get("negative_ratio"), name="negative_ratio", minimum=1
            ),
            minimum_negatives_per_instance=integer(
                payload.get("minimum_negatives_per_instance"),
                name="minimum_negatives_per_instance",
                minimum=1,
            ),
            patience=integer(payload.get("patience"), name="patience", minimum=1),
            warm_start_budget=integer(
                payload.get("warm_start_budget"), name="warm_start_budget", minimum=0
            ),
            bootstrap_draws=integer(
                payload.get("bootstrap_draws"), name="bootstrap_draws", minimum=1
            ),
            bootstrap_seed=integer(
                payload.get("bootstrap_seed"), name="bootstrap_seed", minimum=0
            ),
            model_seed=integer(payload.get("model_seed"), name="model_seed", minimum=0),
            scenarios=tuple(scenarios),
        )


@dataclass(frozen=True, slots=True)
class ResearchReport:
    config: dict[str, object]
    config_fingerprint: str
    training_dataset_metadata: dict[str, object]
    validation_dataset_metadata: dict[str, object]
    invariant_training: TrainingReport
    binding_training: TrainingReport
    evaluations: tuple[EvaluationReport, ...]
    checkpoints: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "training_dataset_metadata": self.training_dataset_metadata,
            "validation_dataset_metadata": self.validation_dataset_metadata,
            "invariant_training": self.invariant_training.to_dict(),
            "binding_training": self.binding_training.to_dict(),
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "checkpoints": self.checkpoints,
        }


def _integer_list(payload: dict[str, object], name: str) -> list[int]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in value
    ):
        raise ValueError(f"research field {name!r} must be an integer array")
    return list(cast(tuple[int, ...] | list[int], value))


def _string_list(payload: dict[str, object], name: str) -> list[str]:
    value = payload.get(name)
    if not isinstance(value, (list, tuple)) or not all(isinstance(entry, str) for entry in value):
        raise ValueError(f"research field {name!r} must be a string array")
    return list(cast(tuple[str, ...] | list[str], value))


def _training_datasets(config: ResearchConfig) -> tuple[WarmStartDataset, WarmStartDataset]:
    training = generate_dataset(
        count=config.training_count,
        node_counts=config.training_node_counts,
        regimes=config.training_regimes,
        seed=config.training_seed,
    )
    validation = generate_dataset(
        count=config.validation_count,
        node_counts=config.validation_node_counts,
        regimes=config.validation_regimes,
        seed=config.validation_seed,
    )
    return training, validation


def run_research(
    config: ResearchConfig,
    *,
    checkpoint_directory: str | Path,
) -> ResearchReport:
    """Train invariant/binding targets once and evaluate fixed seeds under every shift."""

    training, validation = _training_datasets(config)
    architecture = ConstraintScorerConfig(
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
    )

    def training_config(seed: int) -> TrainingConfig:
        return TrainingConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            negative_ratio=config.negative_ratio,
            minimum_negatives_per_instance=config.minimum_negatives_per_instance,
            validation_budget=config.warm_start_budget,
            patience=config.patience,
            seed=seed,
        )

    invariant_model, invariant_report = train_constraint_scorer(
        training,
        validation,
        target="invariant",
        model_config=architecture,
        training_config=training_config(config.model_seed),
    )
    binding_model, binding_report = train_constraint_scorer(
        training,
        validation,
        target="binding",
        model_config=architecture,
        training_config=training_config(config.model_seed + 1),
    )
    checkpoint_root = Path(checkpoint_directory)
    invariant_path = checkpoint_root / "invariant-scorer.safetensors"
    binding_path = checkpoint_root / "binding-scorer.safetensors"
    save_checkpoint(
        invariant_model,
        invariant_path,
        target="invariant",
        metadata={"training_report": invariant_report.to_dict()},
    )
    save_checkpoint(
        binding_model,
        binding_path,
        target="binding",
        metadata={"training_report": binding_report.to_dict()},
    )

    evaluations: list[EvaluationReport] = []
    for offset, scenario in enumerate(config.scenarios):
        dataset = generate_dataset(
            count=scenario.count,
            node_counts=(scenario.node_count,),
            regimes=(scenario.regime,),
            seed=scenario.seed,
        )
        evaluations.append(
            evaluate_dataset(
                dataset,
                scenario=scenario.name,
                budget=config.warm_start_budget,
                invariant_model=invariant_model,
                binding_model=binding_model,
                random_seed=config.bootstrap_seed + 10_000 * offset,
                bootstrap_seed=config.bootstrap_seed + 100_000 * offset,
                bootstrap_draws=config.bootstrap_draws,
            )
        )
    config_payload = config.to_dict()
    return ResearchReport(
        config=config_payload,
        config_fingerprint=sha256_json(config_payload),
        training_dataset_metadata=training.to_metadata(),
        validation_dataset_metadata=validation.to_metadata(),
        invariant_training=invariant_report,
        binding_training=binding_report,
        evaluations=tuple(evaluations),
        checkpoints={
            "invariant": str(invariant_path),
            "binding": str(binding_path),
        },
    )
