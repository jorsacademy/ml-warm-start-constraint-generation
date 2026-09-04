from __future__ import annotations

import math
from pathlib import Path

from warmcg.dataset import WarmStartDataset
from warmcg.evaluation import evaluate_dataset, save_report_csv, save_report_json
from warmcg.model import ConstraintScorer
from warmcg.solver import run_constraint_generation
from warmcg.training import TrainingReport
from warmcg.warmstart import select_warm_start


def test_random_and_compactness_selectors_respect_budget(
    small_validation_dataset: WarmStartDataset,
) -> None:
    record = small_validation_dataset.records[0]
    random_first = select_warm_start(
        record.instance,
        method="random",
        budget=5,
        seed=123,
        record=record,
    )
    random_second = select_warm_start(
        record.instance,
        method="random",
        budget=5,
        seed=123,
        record=record,
    )
    compactness = select_warm_start(
        record.instance,
        method="compactness",
        budget=5,
        record=record,
    )
    assert random_first.cuts == random_second.cuts
    assert len(random_first.cuts) == 5
    assert len(compactness.cuts) == 5
    assert compactness.invariant_overlap is not None


def test_oracle_invariant_is_one_shot(
    small_validation_dataset: WarmStartDataset,
) -> None:
    for record in small_validation_dataset.records:
        selection = select_warm_start(
            record.instance,
            method="oracle_invariant_full",
            budget=1,
            record=record,
        )
        result = run_constraint_generation(record.instance, initial_cuts=selection.cuts)
        assert result.one_shot
        assert math.isclose(result.solution.objective, record.exact_tour.objective, abs_tol=1e-9)


def test_evaluation_preserves_exactness_and_matched_budgets(
    small_validation_dataset: WarmStartDataset,
    trained_invariant: tuple[ConstraintScorer, TrainingReport],
    trained_binding: tuple[ConstraintScorer, TrainingReport],
) -> None:
    invariant_model, _ = trained_invariant
    binding_model, _ = trained_binding
    report = evaluate_dataset(
        small_validation_dataset,
        scenario="test",
        budget=4,
        invariant_model=invariant_model,
        binding_model=binding_model,
        bootstrap_draws=20,
    )
    assert report.metadata["all_results_certified"] is True
    assert report.metadata["all_results_held_karp_verified"] is True
    assert len(report.summary_rows) == 8
    for row in report.summary_rows:
        assert row.certification_rate == 1.0
        assert row.held_karp_verification_rate == 1.0
        if row.method not in {
            "cold",
            "oracle_invariant_full",
            "oracle_trajectory",
            "oracle_invariant_matched",
        }:
            assert math.isclose(row.mean_preloaded_cut_count, 4.0)
    oracle = next(row for row in report.summary_rows if row.method == "oracle_invariant_full")
    assert oracle.one_shot_rate == 1.0


def test_evaluation_serializes_json_and_csv(
    tmp_path: Path,
    small_validation_dataset: WarmStartDataset,
    trained_invariant: tuple[ConstraintScorer, TrainingReport],
    trained_binding: tuple[ConstraintScorer, TrainingReport],
) -> None:
    report = evaluate_dataset(
        small_validation_dataset,
        scenario="serialization",
        budget=3,
        invariant_model=trained_invariant[0],
        binding_model=trained_binding[0],
        bootstrap_draws=10,
    )
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"
    save_report_json(report, json_path)
    save_report_csv(report, csv_path)
    assert json_path.read_text(encoding="utf-8").startswith("{")
    assert "mean_master_solve_count" in csv_path.read_text(encoding="utf-8")
