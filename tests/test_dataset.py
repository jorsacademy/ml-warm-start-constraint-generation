from __future__ import annotations

import json
from pathlib import Path

import pytest

from warmcg.dataset import (
    generate_dataset,
    load_dataset,
    save_dataset,
    target_label_vector,
)
from warmcg.domain import enumerate_candidate_cuts


def test_dataset_round_trip_and_exact_verification(tmp_path: Path) -> None:
    dataset = generate_dataset(
        count=3,
        node_counts=(6, 7),
        regimes=("uniform", "strongly_clustered"),
        seed=20,
    )
    path = tmp_path / "dataset.jsonl"
    save_dataset(dataset, path)
    loaded = load_dataset(path)
    assert loaded.fingerprint == dataset.fingerprint
    assert loaded.to_metadata() == dataset.to_metadata()


def test_dataset_rejects_tampering(tmp_path: Path) -> None:
    dataset = generate_dataset(
        count=2,
        node_counts=(7,),
        regimes=("strongly_clustered",),
        seed=30,
    )
    path = tmp_path / "dataset.jsonl"
    save_dataset(dataset, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["instance"]["coordinates"][0][0] += 0.01
    lines[1] = json.dumps(record, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path, verify=False)


def test_target_vectors_match_record_sets() -> None:
    dataset = generate_dataset(
        count=1,
        node_counts=(8,),
        regimes=("strongly_clustered",),
        seed=2,
    )
    record = dataset.records[0]
    candidates = enumerate_candidate_cuts(record.instance.node_count)
    for target in ("invariant", "trajectory", "binding", "initial"):
        labels = target_label_vector(record, candidates, target)
        assert int(labels.sum()) == len(record.cuts_for_target(target))


def test_invariant_labels_are_subset_of_trajectory_labels() -> None:
    dataset = generate_dataset(
        count=4,
        node_counts=(7, 8),
        regimes=("clustered", "strongly_clustered"),
        seed=40,
    )
    for record in dataset.records:
        assert {cut.nodes for cut in record.invariant_cuts}.issubset(
            {cut.nodes for cut in record.trajectory_cuts}
        )
