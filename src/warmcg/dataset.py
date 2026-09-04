"""Exact-labeled TSP corpora for constraint warm-start learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from warmcg.domain import (
    CutConstraint,
    TourSolution,
    TSPInstance,
    cut_value_on_tour,
    enumerate_candidate_cuts,
    generate_instance,
    make_tour_solution,
    solve_held_karp,
)
from warmcg.solver import build_one_shot_core, run_constraint_generation, solve_master
from warmcg.utils import as_object_dict, integer, sha256_json, string

DATASET_SCHEMA_VERSION = "warmcg-tsp-sec-v1"
LabelTarget = Literal["invariant", "trajectory", "binding", "initial"]


@dataclass(frozen=True, slots=True)
class LabeledTSPRecord:
    """One instance with exact tour and several constraint-set supervision targets."""

    instance: TSPInstance
    exact_tour: TourSolution
    cold_base_objective: float
    cold_master_solve_count: int
    trajectory_cuts: tuple[CutConstraint, ...]
    initial_cuts: tuple[CutConstraint, ...]
    invariant_cuts: tuple[CutConstraint, ...]
    binding_cuts: tuple[CutConstraint, ...]
    invariant_reduction_solve_count: int
    candidate_count: int

    def __post_init__(self) -> None:
        if self.cold_master_solve_count <= 0:
            raise ValueError("cold_master_solve_count must be positive")
        if self.invariant_reduction_solve_count < 0:
            raise ValueError("invariant_reduction_solve_count must be nonnegative")
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        for collection in (
            self.trajectory_cuts,
            self.initial_cuts,
            self.invariant_cuts,
            self.binding_cuts,
        ):
            if len({cut.nodes for cut in collection}) != len(collection):
                raise ValueError("record contains duplicate cuts")
            if any(cut.node_count != self.instance.node_count for cut in collection):
                raise ValueError("record cut node count does not match the instance")
        trajectory = {cut.nodes for cut in self.trajectory_cuts}
        if not {cut.nodes for cut in self.initial_cuts}.issubset(trajectory):
            raise ValueError("initial cuts must be a subset of trajectory cuts")
        if not {cut.nodes for cut in self.invariant_cuts}.issubset(trajectory):
            raise ValueError("invariant cuts must be a subset of trajectory cuts")

    def cuts_for_target(self, target: LabelTarget) -> tuple[CutConstraint, ...]:
        if target == "invariant":
            return self.invariant_cuts
        if target == "trajectory":
            return self.trajectory_cuts
        if target == "binding":
            return self.binding_cuts
        if target == "initial":
            return self.initial_cuts
        raise ValueError(f"unsupported label target: {target}")

    def to_dict(self) -> dict[str, object]:
        return {
            "instance": self.instance.to_dict(),
            "exact_tour": self.exact_tour.to_dict(),
            "cold_base_objective": self.cold_base_objective,
            "cold_master_solve_count": self.cold_master_solve_count,
            "trajectory_cuts": [cut.to_dict() for cut in self.trajectory_cuts],
            "initial_cuts": [cut.to_dict() for cut in self.initial_cuts],
            "invariant_cuts": [cut.to_dict() for cut in self.invariant_cuts],
            "binding_cuts": [cut.to_dict() for cut in self.binding_cuts],
            "invariant_reduction_solve_count": self.invariant_reduction_solve_count,
            "candidate_count": self.candidate_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> LabeledTSPRecord:
        raw_instance = as_object_dict(payload.get("instance"), name="record.instance")
        raw_tour = as_object_dict(payload.get("exact_tour"), name="record.exact_tour")
        instance = TSPInstance.from_dict(raw_instance)
        raw_order = raw_tour.get("order")
        if not isinstance(raw_order, list) or not all(
            isinstance(node, int) and not isinstance(node, bool) for node in raw_order
        ):
            raise ValueError("record exact_tour.order must be an integer JSON array")
        exact_tour = make_tour_solution(instance, tuple(raw_order))
        objective = raw_tour.get("objective")
        if isinstance(objective, bool) or not isinstance(objective, (int, float)):
            raise ValueError("record exact_tour.objective must be numeric")
        if not np.isclose(exact_tour.objective, float(objective), rtol=1e-9, atol=1e-9):
            raise ValueError("record exact tour objective is inconsistent")

        def read_cuts(name: str) -> tuple[CutConstraint, ...]:
            raw = payload.get(name)
            if not isinstance(raw, list):
                raise ValueError(f"record {name} must be a JSON array")
            result: list[CutConstraint] = []
            for entry in raw:
                result.append(CutConstraint.from_dict(as_object_dict(entry, name=name)))
            return tuple(result)

        base_objective_raw = payload.get("cold_base_objective")
        if isinstance(base_objective_raw, bool) or not isinstance(base_objective_raw, (int, float)):
            raise ValueError("record cold_base_objective must be numeric")
        return cls(
            instance=instance,
            exact_tour=exact_tour,
            cold_base_objective=float(base_objective_raw),
            cold_master_solve_count=integer(
                payload.get("cold_master_solve_count"),
                name="cold_master_solve_count",
                minimum=1,
            ),
            trajectory_cuts=read_cuts("trajectory_cuts"),
            initial_cuts=read_cuts("initial_cuts"),
            invariant_cuts=read_cuts("invariant_cuts"),
            binding_cuts=read_cuts("binding_cuts"),
            invariant_reduction_solve_count=integer(
                payload.get("invariant_reduction_solve_count"),
                name="invariant_reduction_solve_count",
                minimum=0,
            ),
            candidate_count=integer(
                payload.get("candidate_count"), name="candidate_count", minimum=1
            ),
        )


@dataclass(frozen=True, slots=True)
class WarmStartDataset:
    """A deterministic set of whole-instance constraint-learning records."""

    records: tuple[LabeledTSPRecord, ...]
    fingerprint: str
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("a dataset must contain at least one record")
        if len({record.instance.instance_id for record in self.records}) != len(self.records):
            raise ValueError("dataset instance identifiers must be unique")
        expected = dataset_fingerprint(self.records, self.metadata)
        if expected != self.fingerprint:
            raise ValueError("dataset fingerprint does not match its records")

    @property
    def node_counts(self) -> tuple[int, ...]:
        return tuple(sorted({record.instance.node_count for record in self.records}))

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted({record.instance.regime for record in self.records}))

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_count": len(self.records),
            "fingerprint": self.fingerprint,
            "node_counts": list(self.node_counts),
            "regimes": list(self.regimes),
            "metadata": self.metadata,
        }


def _cut_payload(cuts: tuple[CutConstraint, ...]) -> list[list[int]]:
    return [list(cut.nodes) for cut in sorted(cuts)]


def dataset_fingerprint(
    records: tuple[LabeledTSPRecord, ...],
    metadata: dict[str, object],
) -> str:
    """Hash mathematical content while excluding runtime measurements."""

    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "metadata": metadata,
        "records": [
            {
                "instance": record.instance.to_dict(),
                "exact_tour": {
                    "order": list(record.exact_tour.order),
                    "objective": record.exact_tour.objective,
                },
                "cold_base_objective": record.cold_base_objective,
                "cold_master_solve_count": record.cold_master_solve_count,
                "trajectory_cuts": _cut_payload(record.trajectory_cuts),
                "initial_cuts": _cut_payload(record.initial_cuts),
                "invariant_cuts": _cut_payload(record.invariant_cuts),
                "binding_cuts": _cut_payload(record.binding_cuts),
                "candidate_count": record.candidate_count,
            }
            for record in records
        ],
    }
    return sha256_json(payload)


def label_record(
    instance: TSPInstance,
    *,
    held_karp_maximum_nodes: int = 14,
) -> LabeledTSPRecord:
    """Solve one instance offline and construct trajectory, core, and binding labels."""

    if instance.node_count > held_karp_maximum_nodes:
        raise ValueError("offline labeling requires Held-Karp verification for this benchmark")
    cold = run_constraint_generation(
        instance,
        verify_with_held_karp=True,
        held_karp_maximum_nodes=held_karp_maximum_nodes,
    )
    oracle = solve_held_karp(instance, maximum_nodes=held_karp_maximum_nodes)
    if not np.isclose(cold.solution.objective, oracle.objective, rtol=1e-9, atol=1e-9):
        raise RuntimeError("cold constraint generation disagrees with Held-Karp")
    core = build_one_shot_core(
        instance,
        cold.generated_cuts,
        optimum_objective=oracle.objective,
    )
    candidates = enumerate_candidate_cuts(instance.node_count)
    binding = tuple(cut for cut in candidates if cut_value_on_tour(cut, cold.solution) == 2)
    initial = cold.iterations[0].generated_cuts if cold.iterations else ()
    return LabeledTSPRecord(
        instance=instance,
        exact_tour=cold.solution,
        cold_base_objective=cold.iterations[0].master_objective,
        cold_master_solve_count=cold.master_solve_count,
        trajectory_cuts=cold.generated_cuts,
        initial_cuts=initial,
        invariant_cuts=core.cuts,
        binding_cuts=binding,
        invariant_reduction_solve_count=core.master_solve_count,
        candidate_count=len(candidates),
    )


def generate_dataset(
    *,
    count: int,
    node_counts: tuple[int, ...],
    regimes: tuple[str, ...],
    seed: int,
    held_karp_maximum_nodes: int = 14,
) -> WarmStartDataset:
    """Generate and exactly label a whole-instance corpus."""

    if count <= 0 or not node_counts or not regimes:
        raise ValueError("count, node_counts, and regimes must be nonempty")
    if any(node_count < 5 or node_count > held_karp_maximum_nodes for node_count in node_counts):
        raise ValueError("node_counts fall outside the exact-labeling range")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    records: list[LabeledTSPRecord] = []
    for index in range(count):
        node_count = node_counts[index % len(node_counts)]
        regime = regimes[index % len(regimes)]
        instance_seed = seed + index
        instance = generate_instance(
            node_count=node_count,
            regime=regime,
            seed=instance_seed,
            instance_id=f"record-{index:05d}-{regime}-n{node_count}-seed{instance_seed}",
        )
        records.append(label_record(instance, held_karp_maximum_nodes=held_karp_maximum_nodes))
    metadata: dict[str, object] = {
        "generation_seed": seed,
        "requested_node_counts": list(node_counts),
        "requested_regimes": list(regimes),
        "held_karp_maximum_nodes": held_karp_maximum_nodes,
    }
    record_tuple = tuple(records)
    return WarmStartDataset(
        records=record_tuple,
        fingerprint=dataset_fingerprint(record_tuple, metadata),
        metadata=metadata,
    )


def save_dataset(dataset: WarmStartDataset, path: str | Path) -> None:
    """Write a manifest followed by exact-labeled records as JSON Lines."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dataset.to_metadata(), sort_keys=True, allow_nan=False) + "\n")
        for record in dataset.records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False) + "\n")


def _verify_record(record: LabeledTSPRecord, *, held_karp_maximum_nodes: int) -> None:
    instance = record.instance
    if instance.node_count > held_karp_maximum_nodes:
        raise ValueError("record exceeds the configured verification limit")
    oracle = solve_held_karp(instance, maximum_nodes=held_karp_maximum_nodes)
    if not np.isclose(record.exact_tour.objective, oracle.objective, rtol=1e-9, atol=1e-9):
        raise ValueError("stored exact tour is not Held-Karp optimal")
    candidate_count = len(enumerate_candidate_cuts(instance.node_count))
    if candidate_count != record.candidate_count:
        raise ValueError("stored candidate count is inconsistent")
    trajectory_master = solve_master(instance, record.trajectory_cuts)
    if not trajectory_master.connected or not np.isclose(
        trajectory_master.objective,
        oracle.objective,
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("stored trajectory cuts do not recover an optimal tour")
    invariant_master = solve_master(instance, record.invariant_cuts)
    if not invariant_master.connected or not np.isclose(
        invariant_master.objective,
        oracle.objective,
        rtol=1e-9,
        atol=1e-9,
    ):
        raise ValueError("stored invariant cuts do not recover an optimal tour")
    if any(cut_value_on_tour(cut, record.exact_tour) != 2 for cut in record.binding_cuts):
        raise ValueError("stored binding label is not tight at the exact tour")


def load_dataset(
    path: str | Path,
    *,
    verify: bool = True,
) -> WarmStartDataset:
    """Load a corpus, recompute its digest, and optionally re-audit exact labels."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("dataset must contain a manifest and at least one record")
    manifest_raw = json.loads(lines[0])
    manifest = as_object_dict(manifest_raw, name="dataset manifest")
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema version")
    raw_metadata = manifest.get("metadata")
    metadata = cast(dict[str, object], as_object_dict(raw_metadata, name="dataset metadata"))
    records: list[LabeledTSPRecord] = []
    for line_number, line in enumerate(lines[1:], start=2):
        raw_record = json.loads(line)
        records.append(
            LabeledTSPRecord.from_dict(
                as_object_dict(raw_record, name=f"dataset record at line {line_number}")
            )
        )
    record_tuple = tuple(records)
    fingerprint = string(manifest.get("fingerprint"), name="dataset fingerprint")
    dataset = WarmStartDataset(record_tuple, fingerprint, metadata)
    if integer(manifest.get("record_count"), name="record_count", minimum=1) != len(records):
        raise ValueError("dataset manifest record count is inconsistent")
    if verify:
        limit_raw = metadata.get("held_karp_maximum_nodes", 14)
        limit = integer(limit_raw, name="held_karp_maximum_nodes", minimum=5)
        for record in records:
            _verify_record(record, held_karp_maximum_nodes=limit)
    return dataset


def target_label_vector(
    record: LabeledTSPRecord,
    candidates: tuple[CutConstraint, ...],
    target: LabelTarget,
) -> np.ndarray:
    """Return a Boolean label vector aligned with a candidate cut universe."""

    positives = {cut.nodes for cut in record.cuts_for_target(target)}
    labels = np.asarray([cut.nodes in positives for cut in candidates], dtype=np.float32)
    return labels
