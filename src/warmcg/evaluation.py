"""Solver-grounded evaluation of constraint preload policies."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np

from warmcg.dataset import LabeledTSPRecord, WarmStartDataset
from warmcg.model import ConstraintScorer
from warmcg.solver import ConstraintGenerationResult, run_constraint_generation
from warmcg.utils import write_json
from warmcg.warmstart import (
    SetOverlap,
    WarmStartMethod,
    WarmStartSelection,
    select_warm_start,
)

DEFAULT_METHODS: tuple[WarmStartMethod, ...] = (
    "cold",
    "random",
    "compactness",
    "binding_model",
    "invariant_model",
    "oracle_trajectory",
    "oracle_invariant_matched",
    "oracle_invariant_full",
)


@dataclass(frozen=True, slots=True)
class InstanceMethodResult:
    instance_id: str
    regime: str
    node_count: int
    method: str
    requested_budget: int
    preloaded_cut_count: int
    online_generated_cut_count: int
    final_active_cut_count: int
    master_solve_count: int
    one_shot: bool
    first_master_objective: float
    exact_objective: float
    root_gap_closure: float | None
    total_mip_nodes: int
    selection_seconds: float
    solve_seconds: float
    total_runtime_seconds: float
    certified: bool
    held_karp_verified: bool
    invariant_precision: float | None
    invariant_recall: float | None
    trajectory_precision: float | None
    trajectory_recall: float | None
    binding_precision: float | None
    binding_recall: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MethodMetrics:
    scenario: str
    method: str
    instance_count: int
    requested_budget: int
    mean_preloaded_cut_count: float
    mean_online_generated_cut_count: float
    mean_final_active_cut_count: float
    mean_master_solve_count: float
    median_master_solve_count: float
    p90_master_solve_count: float
    one_shot_rate: float
    mean_root_gap_closure: float | None
    mean_total_mip_nodes: float
    mean_selection_seconds: float
    mean_solve_seconds: float
    mean_total_runtime_seconds: float
    certification_rate: float
    held_karp_verification_rate: float
    mean_invariant_precision: float | None
    mean_invariant_recall: float | None
    mean_trajectory_precision: float | None
    mean_trajectory_recall: float | None
    mean_binding_precision: float | None
    mean_binding_recall: float | None
    mean_solve_reduction_vs_cold: float
    solve_reduction_ci_low: float
    solve_reduction_ci_high: float
    mean_online_cut_reduction_vs_cold: float
    online_cut_reduction_ci_low: float
    online_cut_reduction_ci_high: float
    mean_runtime_difference_vs_cold: float
    runtime_difference_ci_low: float
    runtime_difference_ci_high: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    scenario: str
    budget: int
    summary_rows: tuple[MethodMetrics, ...]
    instance_rows: tuple[InstanceMethodResult, ...]
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "budget": self.budget,
            "summary_rows": [row.to_dict() for row in self.summary_rows],
            "instance_rows": [row.to_dict() for row in self.instance_rows],
            "metadata": self.metadata,
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    return float(np.percentile(np.asarray(values, dtype=float), q, method="linear"))


def _mean_optional(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return float(np.mean(np.asarray(observed, dtype=float)))


def _bootstrap_interval(
    values: list[float],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    if not values or draws <= 0:
        raise ValueError("bootstrap requires values and a positive draw count")
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        indices = rng.integers(0, data.size, size=data.size)
        means[draw] = float(np.mean(data[indices]))
    return (
        float(np.percentile(means, 2.5, method="linear")),
        float(np.percentile(means, 97.5, method="linear")),
    )


def _root_gap_closure(
    base_objective: float,
    first_objective: float,
    exact_objective: float,
) -> float | None:
    denominator = exact_objective - base_objective
    if denominator <= 1e-10:
        return None
    value = (first_objective - base_objective) / denominator
    if value < -1e-8 or value > 1.0 + 1e-7:
        raise RuntimeError("warm-start master bound lies outside the valid relaxation interval")
    return min(1.0, max(0.0, value))


def _overlap_value(
    overlap: SetOverlap | None,
    field: Literal["precision", "recall"],
) -> float | None:
    if overlap is None:
        return None
    return overlap.precision if field == "precision" else overlap.recall


def _instance_result(
    record: LabeledTSPRecord,
    selection: WarmStartSelection,
    result: ConstraintGenerationResult,
) -> InstanceMethodResult:
    if not math.isclose(
        result.solution.objective,
        record.exact_tour.objective,
        rel_tol=1e-8,
        abs_tol=1e-8,
    ):
        raise RuntimeError("warm-started constraint generation returned a nonoptimal tour")
    if not result.certified:
        raise RuntimeError(
            "benchmark refuses to record an uncertified constraint-generation result"
        )
    first_objective = result.iterations[0].master_objective
    return InstanceMethodResult(
        instance_id=record.instance.instance_id,
        regime=record.instance.regime,
        node_count=record.instance.node_count,
        method=selection.method,
        requested_budget=selection.budget,
        preloaded_cut_count=len(selection.cuts),
        online_generated_cut_count=result.online_generated_cut_count,
        final_active_cut_count=len(result.active_cuts),
        master_solve_count=result.master_solve_count,
        one_shot=result.one_shot,
        first_master_objective=first_objective,
        exact_objective=record.exact_tour.objective,
        root_gap_closure=_root_gap_closure(
            record.cold_base_objective,
            first_objective,
            record.exact_tour.objective,
        ),
        total_mip_nodes=result.total_mip_nodes,
        selection_seconds=selection.selection_seconds,
        solve_seconds=result.total_solve_seconds,
        total_runtime_seconds=selection.selection_seconds + result.total_runtime_seconds,
        certified=result.certified,
        held_karp_verified=result.held_karp_verified,
        invariant_precision=_overlap_value(selection.invariant_overlap, "precision"),
        invariant_recall=_overlap_value(selection.invariant_overlap, "recall"),
        trajectory_precision=_overlap_value(selection.trajectory_overlap, "precision"),
        trajectory_recall=_overlap_value(selection.trajectory_overlap, "recall"),
        binding_precision=_overlap_value(selection.binding_overlap, "precision"),
        binding_recall=_overlap_value(selection.binding_overlap, "recall"),
    )


def evaluate_dataset(
    dataset: WarmStartDataset,
    *,
    scenario: str,
    budget: int,
    invariant_model: ConstraintScorer | None = None,
    binding_model: ConstraintScorer | None = None,
    methods: tuple[WarmStartMethod, ...] = DEFAULT_METHODS,
    random_seed: int = 0,
    bootstrap_seed: int = 0,
    bootstrap_draws: int = 500,
) -> EvaluationReport:
    """Run every preload policy through the same exact constraint-generation loop."""

    if not scenario:
        raise ValueError("scenario must be nonempty")
    if budget < 0 or bootstrap_draws <= 0:
        raise ValueError("budget and bootstrap_draws are invalid")
    if len(set(methods)) != len(methods) or "cold" not in methods:
        raise ValueError("methods must be unique and include cold")
    if "invariant_model" in methods and invariant_model is None:
        raise ValueError("invariant_model method requires a trained model")
    if "binding_model" in methods and binding_model is None:
        raise ValueError("binding_model method requires a trained model")

    instance_rows: list[InstanceMethodResult] = []
    for record_index, record in enumerate(dataset.records):
        for method_index, method in enumerate(methods):
            model = (
                invariant_model
                if method == "invariant_model"
                else binding_model
                if method == "binding_model"
                else None
            )
            selection = select_warm_start(
                record.instance,
                method=method,
                budget=budget,
                seed=random_seed + 10_000 * record_index + method_index,
                model=model,
                record=record,
            )
            result = run_constraint_generation(
                record.instance,
                initial_cuts=selection.cuts,
                verify_with_held_karp=True,
                held_karp_maximum_nodes=max(dataset.node_counts),
            )
            instance_rows.append(_instance_result(record, selection, result))

    by_method: dict[str, list[InstanceMethodResult]] = {method: [] for method in methods}
    for row in instance_rows:
        by_method[row.method].append(row)
    cold_rows = by_method["cold"]
    cold_by_instance = {row.instance_id: row for row in cold_rows}
    summary: list[MethodMetrics] = []
    for method_index, method in enumerate(methods):
        rows = by_method[method]
        solves = [float(row.master_solve_count) for row in rows]
        solve_reductions = [
            float(cold_by_instance[row.instance_id].master_solve_count - row.master_solve_count)
            for row in rows
        ]
        online_cut_reductions = [
            float(
                cold_by_instance[row.instance_id].online_generated_cut_count
                - row.online_generated_cut_count
            )
            for row in rows
        ]
        runtime_differences = [
            row.total_runtime_seconds - cold_by_instance[row.instance_id].total_runtime_seconds
            for row in rows
        ]
        solve_ci = _bootstrap_interval(
            solve_reductions,
            seed=bootstrap_seed + 100 * method_index,
            draws=bootstrap_draws,
        )
        cut_ci = _bootstrap_interval(
            online_cut_reductions,
            seed=bootstrap_seed + 100 * method_index + 1,
            draws=bootstrap_draws,
        )
        runtime_ci = _bootstrap_interval(
            runtime_differences,
            seed=bootstrap_seed + 100 * method_index + 2,
            draws=bootstrap_draws,
        )
        summary.append(
            MethodMetrics(
                scenario=scenario,
                method=method,
                instance_count=len(rows),
                requested_budget=budget,
                mean_preloaded_cut_count=float(np.mean([row.preloaded_cut_count for row in rows])),
                mean_online_generated_cut_count=float(
                    np.mean([row.online_generated_cut_count for row in rows])
                ),
                mean_final_active_cut_count=float(
                    np.mean([row.final_active_cut_count for row in rows])
                ),
                mean_master_solve_count=float(np.mean(solves)),
                median_master_solve_count=float(median(solves)),
                p90_master_solve_count=_percentile(solves, 90.0),
                one_shot_rate=float(np.mean([row.one_shot for row in rows])),
                mean_root_gap_closure=_mean_optional(row.root_gap_closure for row in rows),
                mean_total_mip_nodes=float(np.mean([row.total_mip_nodes for row in rows])),
                mean_selection_seconds=float(np.mean([row.selection_seconds for row in rows])),
                mean_solve_seconds=float(np.mean([row.solve_seconds for row in rows])),
                mean_total_runtime_seconds=float(
                    np.mean([row.total_runtime_seconds for row in rows])
                ),
                certification_rate=float(np.mean([row.certified for row in rows])),
                held_karp_verification_rate=float(
                    np.mean([row.held_karp_verified for row in rows])
                ),
                mean_invariant_precision=_mean_optional(row.invariant_precision for row in rows),
                mean_invariant_recall=_mean_optional(row.invariant_recall for row in rows),
                mean_trajectory_precision=_mean_optional(row.trajectory_precision for row in rows),
                mean_trajectory_recall=_mean_optional(row.trajectory_recall for row in rows),
                mean_binding_precision=_mean_optional(row.binding_precision for row in rows),
                mean_binding_recall=_mean_optional(row.binding_recall for row in rows),
                mean_solve_reduction_vs_cold=float(np.mean(solve_reductions)),
                solve_reduction_ci_low=solve_ci[0],
                solve_reduction_ci_high=solve_ci[1],
                mean_online_cut_reduction_vs_cold=float(np.mean(online_cut_reductions)),
                online_cut_reduction_ci_low=cut_ci[0],
                online_cut_reduction_ci_high=cut_ci[1],
                mean_runtime_difference_vs_cold=float(np.mean(runtime_differences)),
                runtime_difference_ci_low=runtime_ci[0],
                runtime_difference_ci_high=runtime_ci[1],
            )
        )

    return EvaluationReport(
        scenario=scenario,
        budget=budget,
        summary_rows=tuple(summary),
        instance_rows=tuple(instance_rows),
        metadata={
            "dataset_fingerprint": dataset.fingerprint,
            "node_counts": list(dataset.node_counts),
            "regimes": list(dataset.regimes),
            "methods": list(methods),
            "random_seed": random_seed,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_draws": bootstrap_draws,
            "all_results_certified": all(row.certified for row in instance_rows),
            "all_results_held_karp_verified": all(row.held_karp_verified for row in instance_rows),
            "oracle_invariant_budget_note": (
                "oracle_invariant_full preloads the full offline core and is not "
                "cardinality matched"
            ),
            "claims_boundary": (
                "Learning changes only the initial valid SEC set; exact integer separation and "
                "global master solves remain responsible for feasibility and optimality."
            ),
        },
    )


def save_report_json(report: EvaluationReport, path: str | Path) -> None:
    write_json(report.to_dict(), path)


def save_report_csv(report: EvaluationReport, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [row.to_dict() for row in report.summary_rows]
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
