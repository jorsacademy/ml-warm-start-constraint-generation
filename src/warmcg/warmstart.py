"""Constraint preload policies for exact warm-started constraint generation."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from warmcg.dataset import LabeledTSPRecord
from warmcg.domain import (
    CutConstraint,
    TSPInstance,
    candidate_cut_count,
    enumerate_candidate_cuts,
)
from warmcg.features import (
    candidate_features,
    compactness_heuristic_scores,
    rank_top_k,
)
from warmcg.model import ConstraintScorer, score_feature_matrix

WarmStartMethod = Literal[
    "cold",
    "random",
    "compactness",
    "invariant_model",
    "binding_model",
    "oracle_invariant_matched",
    "oracle_invariant_full",
    "oracle_trajectory",
]


@dataclass(frozen=True, slots=True)
class SetOverlap:
    selected_count: int
    reference_count: int
    intersection_count: int
    precision: float | None
    recall: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WarmStartSelection:
    method: str
    budget: int
    candidate_count: int
    cuts: tuple[CutConstraint, ...]
    selection_seconds: float
    invariant_overlap: SetOverlap | None
    trajectory_overlap: SetOverlap | None
    binding_overlap: SetOverlap | None

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "budget": self.budget,
            "candidate_count": self.candidate_count,
            "cuts": [cut.to_dict() for cut in self.cuts],
            "selection_seconds": self.selection_seconds,
            "invariant_overlap": (
                None if self.invariant_overlap is None else self.invariant_overlap.to_dict()
            ),
            "trajectory_overlap": (
                None if self.trajectory_overlap is None else self.trajectory_overlap.to_dict()
            ),
            "binding_overlap": (
                None if self.binding_overlap is None else self.binding_overlap.to_dict()
            ),
        }


def _overlap(
    selected: tuple[CutConstraint, ...],
    reference: tuple[CutConstraint, ...],
) -> SetOverlap:
    selected_nodes = {cut.nodes for cut in selected}
    reference_nodes = {cut.nodes for cut in reference}
    intersection = len(selected_nodes & reference_nodes)
    return SetOverlap(
        selected_count=len(selected_nodes),
        reference_count=len(reference_nodes),
        intersection_count=intersection,
        precision=intersection / len(selected_nodes) if selected_nodes else None,
        recall=intersection / len(reference_nodes) if reference_nodes else None,
    )


def _selection(
    *,
    method: str,
    budget: int,
    candidate_count: int,
    cuts: tuple[CutConstraint, ...],
    selection_seconds: float,
    record: LabeledTSPRecord | None,
) -> WarmStartSelection:
    return WarmStartSelection(
        method=method,
        budget=budget,
        candidate_count=candidate_count,
        cuts=cuts,
        selection_seconds=selection_seconds,
        invariant_overlap=None if record is None else _overlap(cuts, record.invariant_cuts),
        trajectory_overlap=None if record is None else _overlap(cuts, record.trajectory_cuts),
        binding_overlap=None if record is None else _overlap(cuts, record.binding_cuts),
    )


def select_warm_start(
    instance: TSPInstance,
    *,
    method: WarmStartMethod,
    budget: int,
    seed: int = 0,
    model: ConstraintScorer | None = None,
    record: LabeledTSPRecord | None = None,
) -> WarmStartSelection:
    """Select only mathematically valid SECs; exact CG remains responsible for completion."""

    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if method in {"invariant_model", "binding_model"} and model is None:
        raise ValueError(f"method {method!r} requires a trained model")
    if method.startswith("oracle_") and record is None:
        raise ValueError(f"method {method!r} requires exact offline labels")
    if record is not None and record.instance.instance_id != instance.instance_id:
        raise ValueError("warm-start record does not correspond to the supplied instance")

    start = time.perf_counter()
    if method == "cold" or (budget == 0 and method != "oracle_invariant_full"):
        chosen: tuple[CutConstraint, ...] = ()
        return _selection(
            method=method,
            budget=budget,
            candidate_count=candidate_cut_count(instance.node_count),
            cuts=chosen,
            selection_seconds=time.perf_counter() - start,
            record=record,
        )
    candidates = enumerate_candidate_cuts(instance.node_count)
    if method == "random":
        rng = np.random.default_rng(seed)
        count = min(budget, len(candidates))
        indices = np.sort(rng.choice(len(candidates), size=count, replace=False))
        chosen = tuple(candidates[int(index)] for index in indices)
    elif method == "compactness":
        batch = candidate_features(instance, candidates)
        chosen = rank_top_k(
            candidates,
            compactness_heuristic_scores(batch.values).tolist(),
            budget,
        )
    elif method in {"invariant_model", "binding_model"}:
        if model is None:
            raise RuntimeError("learned warm-start selection lost its model")
        batch = candidate_features(instance, candidates)
        scores = score_feature_matrix(model, batch.values)
        chosen = rank_top_k(candidates, scores.tolist(), budget)
    elif method == "oracle_invariant_matched":
        if record is None:
            raise RuntimeError("matched invariant oracle lost its record")
        invariant_nodes = {cut.nodes for cut in record.invariant_cuts}
        ordered = tuple(cut for cut in record.trajectory_cuts if cut.nodes in invariant_nodes)
        chosen = ordered[: min(budget, len(ordered))]
    elif method == "oracle_invariant_full":
        if record is None:
            raise RuntimeError("full invariant oracle lost its record")
        # This is deliberately a full-information ceiling, not a matched-budget baseline.
        chosen = tuple(sorted(record.invariant_cuts))
    elif method == "oracle_trajectory":
        if record is None:
            raise RuntimeError("oracle trajectory selection lost its record")
        chosen = tuple(record.trajectory_cuts[: min(budget, len(record.trajectory_cuts))])
    else:
        raise ValueError(f"unsupported warm-start method: {method}")

    if len({cut.nodes for cut in chosen}) != len(chosen):
        raise RuntimeError("warm-start selector returned duplicate cuts")
    if any(cut.node_count != instance.node_count for cut in chosen):
        raise RuntimeError("warm-start selector returned an incompatible cut")
    elapsed = time.perf_counter() - start
    if not math.isfinite(elapsed):
        raise RuntimeError("warm-start selection runtime is non-finite")
    return _selection(
        method=method,
        budget=budget,
        candidate_count=len(candidates),
        cuts=chosen,
        selection_seconds=elapsed,
        record=record,
    )
