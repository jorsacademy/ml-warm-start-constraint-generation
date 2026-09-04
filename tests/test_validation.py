from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from warmcg.dataset import generate_dataset
from warmcg.domain import TSPInstance, generate_instance
from warmcg.experiment import ResearchConfig, ScenarioConfig
from warmcg.features import FEATURE_NAMES
from warmcg.model import (
    ConstraintScorer,
    ConstraintScorerConfig,
    load_checkpoint,
    score_feature_matrix,
)
from warmcg.utils import as_object_dict, finite_float, integer, string
from warmcg.warmstart import select_warm_start


def test_model_validation_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ConstraintScorerConfig(hidden_dim=0)
    model = ConstraintScorer(ConstraintScorerConfig(hidden_dim=4, hidden_layers=1))
    width = len(FEATURE_NAMES)
    with pytest.raises(ValueError):
        model.set_normalization(np.zeros(width - 1), np.ones(width - 1))
    with pytest.raises(ValueError):
        model.set_normalization(np.full(width, np.nan), np.ones(width))
    with pytest.raises(ValueError):
        model.set_normalization(np.zeros(width), np.zeros(width))
    with pytest.raises(ValueError):
        model(torch.zeros(width))
    with pytest.raises(ValueError):
        score_feature_matrix(model, np.zeros((2, width)), batch_size=0)
    with pytest.raises(ValueError):
        score_feature_matrix(model, np.zeros((2, width - 1)))
    bad = np.zeros((2, width), dtype=float)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        score_feature_matrix(model, bad)

    path = tmp_path / "bad-metadata.safetensors"
    tensors = {key: value.detach().clone() for key, value in model.state_dict().items()}
    save_file(
        tensors,
        str(path),
        metadata={
            "checkpoint_schema_version": "wrong",
            "feature_schema_version": "tsp-sec-geometry-v1",
            "feature_names": json.dumps(FEATURE_NAMES),
            "model_type": "constraint_scorer",
            "label_target": "invariant",
            "model_config": json.dumps({"hidden_dim": 4, "hidden_layers": 1}),
            "metadata": "{}",
        },
    )
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path)


def test_warm_start_validation_boundaries() -> None:
    instance = generate_instance(node_count=7, regime="clustered", seed=1)
    with pytest.raises(ValueError):
        select_warm_start(instance, method="random", budget=-1)
    with pytest.raises(ValueError):
        select_warm_start(instance, method="random", budget=2, seed=-1)
    with pytest.raises(ValueError):
        select_warm_start(instance, method="invariant_model", budget=2)
    with pytest.raises(ValueError):
        select_warm_start(instance, method="oracle_invariant_full", budget=2)
    record = generate_dataset(
        count=1,
        node_counts=(7,),
        regimes=("strongly_clustered",),
        seed=10,
    ).records[0]
    with pytest.raises(ValueError):
        select_warm_start(instance, method="oracle_invariant_full", budget=2, record=record)


def _base_research_kwargs() -> dict[str, object]:
    return {
        "training_count": 2,
        "training_node_counts": (7,),
        "training_regimes": ("clustered",),
        "training_seed": 1,
        "validation_count": 1,
        "validation_node_counts": (7,),
        "validation_regimes": ("clustered",),
        "validation_seed": 2,
        "hidden_dim": 4,
        "hidden_layers": 1,
        "epochs": 1,
        "batch_size": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "negative_ratio": 2,
        "minimum_negatives_per_instance": 4,
        "patience": 1,
        "warm_start_budget": 2,
        "bootstrap_draws": 2,
        "bootstrap_seed": 3,
        "model_seed": 4,
        "scenarios": (ScenarioConfig(name="s", count=1, node_count=7, regime="clustered", seed=5),),
    }


def test_research_configuration_validation_boundaries() -> None:
    base = _base_research_kwargs()
    for key, value in (
        ("training_count", 0),
        ("training_node_counts", (4,)),
        ("training_regimes", ()),
        ("hidden_dim", 0),
        ("epochs", 0),
        ("weight_decay", -1.0),
        ("minimum_negatives_per_instance", 0),
        ("warm_start_budget", -1),
        ("scenarios", ()),
    ):
        invalid = dict(base)
        invalid[key] = value
        with pytest.raises(ValueError):
            ResearchConfig(**invalid)  # type: ignore[arg-type]
    payload = ResearchConfig(**base).to_dict()  # type: ignore[arg-type]
    payload["schema_version"] = "wrong"
    with pytest.raises(ValueError):
        ResearchConfig.from_dict(payload)


def test_scalar_validation_helpers() -> None:
    assert finite_float(1, name="x") == 1.0
    assert integer(2, name="x", minimum=1) == 2
    assert string("x", name="x") == "x"
    assert as_object_dict({"x": 1}, name="x") == {"x": 1}
    with pytest.raises(ValueError):
        finite_float(True, name="x")
    with pytest.raises(ValueError):
        finite_float(float("inf"), name="x")
    with pytest.raises(ValueError):
        integer(True, name="x")
    with pytest.raises(ValueError):
        integer(0, name="x", minimum=1)
    with pytest.raises(ValueError):
        string("", name="x")
    with pytest.raises(ValueError):
        as_object_dict([], name="x")


def test_instance_from_dict_rejects_malformed_coordinates() -> None:
    with pytest.raises(ValueError):
        TSPInstance.from_dict({"coordinates": "bad"})
