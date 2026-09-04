from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from warmcg.dataset import WarmStartDataset
from warmcg.domain import enumerate_candidate_cuts
from warmcg.features import FEATURE_NAMES, candidate_features
from warmcg.model import (
    ConstraintScorer,
    ConstraintScorerConfig,
    load_checkpoint,
    save_checkpoint,
    score_feature_matrix,
)
from warmcg.training import PreparedSamples, TrainingReport, evaluate_ranking, prepare_samples


def test_model_forward_and_feature_normalization_are_finite() -> None:
    model = ConstraintScorer(ConstraintScorerConfig(hidden_dim=8, hidden_layers=1))
    features = torch.zeros((5, len(FEATURE_NAMES)), dtype=torch.float32)
    logits = model(features)
    assert logits.shape == (5,)
    assert torch.all(torch.isfinite(logits))


def test_prepare_samples_keeps_positive_labels(
    small_training_dataset: WarmStartDataset,
) -> None:
    samples = prepare_samples(
        small_training_dataset,
        target="invariant",
        negative_ratio=3,
        minimum_negatives_per_instance=8,
        seed=9,
    )
    assert isinstance(samples, PreparedSamples)
    assert samples.positive_count == sum(
        len(record.invariant_cuts) for record in small_training_dataset.records
    )
    assert samples.negative_count > 0
    assert len(set(samples.group_ids.tolist())) == len(small_training_dataset.records)


def test_training_produces_ranking_report(
    trained_invariant: tuple[ConstraintScorer, TrainingReport],
) -> None:
    model, report = trained_invariant
    assert report.best_epoch >= 1
    assert report.training_sample_count > 0
    assert report.validation_sample_count > 0
    assert 0.0 <= report.validation_ranking.average_precision <= 1.0
    assert model.parameter_count > 0


def test_checkpoint_round_trip_preserves_scores(
    tmp_path: Path,
    trained_invariant: tuple[ConstraintScorer, TrainingReport],
    small_validation_dataset: WarmStartDataset,
) -> None:
    model, report = trained_invariant
    path = tmp_path / "model.safetensors"
    save_checkpoint(
        model,
        path,
        target="invariant",
        metadata={"report": report.to_dict()},
    )
    loaded, target, metadata = load_checkpoint(path)
    record = small_validation_dataset.records[0]
    candidates = enumerate_candidate_cuts(record.instance.node_count)
    features = candidate_features(record.instance, candidates).values
    assert target == "invariant"
    assert "report" in metadata
    assert np.allclose(
        score_feature_matrix(model, features),
        score_feature_matrix(loaded, features),
    )


def test_checkpoint_rejects_wrong_tensor_schema(tmp_path: Path) -> None:
    model = ConstraintScorer(ConstraintScorerConfig(hidden_dim=4, hidden_layers=1))
    path = tmp_path / "model.safetensors"
    save_checkpoint(model, path, target="binding")
    data = path.read_bytes()
    path.write_bytes(data[:-8])
    with pytest.raises(Exception):
        load_checkpoint(path)


def test_full_ranking_metrics_are_well_formed(
    trained_binding: tuple[ConstraintScorer, TrainingReport],
    small_validation_dataset: WarmStartDataset,
) -> None:
    model, _ = trained_binding
    metrics = evaluate_ranking(
        model,
        small_validation_dataset,
        target="binding",
        budget=4,
    )
    assert metrics.instance_count == len(small_validation_dataset.records)
    assert metrics.candidate_count > metrics.positive_count > 0
    assert 0.0 <= metrics.top_k_precision <= 1.0
    assert 0.0 <= metrics.top_k_recall <= 1.0
