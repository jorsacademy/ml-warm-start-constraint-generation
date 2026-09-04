from __future__ import annotations

import pytest

from warmcg.dataset import WarmStartDataset, generate_dataset
from warmcg.model import ConstraintScorer, ConstraintScorerConfig
from warmcg.training import TrainingConfig, TrainingReport, train_constraint_scorer


@pytest.fixture(scope="session")
def small_training_dataset() -> WarmStartDataset:
    return generate_dataset(
        count=8,
        node_counts=(7, 8),
        regimes=("clustered", "strongly_clustered"),
        seed=100,
    )


@pytest.fixture(scope="session")
def small_validation_dataset() -> WarmStartDataset:
    return generate_dataset(
        count=4,
        node_counts=(7, 8),
        regimes=("clustered", "strongly_clustered"),
        seed=200,
    )


def _train(
    training: WarmStartDataset,
    validation: WarmStartDataset,
    target: str,
    seed: int,
) -> tuple[ConstraintScorer, TrainingReport]:
    return train_constraint_scorer(
        training,
        validation,
        target=target,  # type: ignore[arg-type]
        model_config=ConstraintScorerConfig(hidden_dim=8, hidden_layers=1),
        training_config=TrainingConfig(
            epochs=2,
            batch_size=64,
            learning_rate=0.002,
            negative_ratio=4,
            minimum_negatives_per_instance=12,
            validation_budget=4,
            patience=2,
            seed=seed,
        ),
    )


@pytest.fixture(scope="session")
def trained_invariant(
    small_training_dataset: WarmStartDataset,
    small_validation_dataset: WarmStartDataset,
) -> tuple[ConstraintScorer, TrainingReport]:
    return _train(small_training_dataset, small_validation_dataset, "invariant", 11)


@pytest.fixture(scope="session")
def trained_binding(
    small_training_dataset: WarmStartDataset,
    small_validation_dataset: WarmStartDataset,
) -> tuple[ConstraintScorer, TrainingReport]:
    return _train(small_training_dataset, small_validation_dataset, "binding", 12)
