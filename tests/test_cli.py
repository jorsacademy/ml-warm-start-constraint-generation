from __future__ import annotations

import json
from pathlib import Path

from warmcg.cli import main


def test_cli_end_to_end(tmp_path: Path, capsys: object) -> None:
    instance = tmp_path / "instance.json"
    oracle = tmp_path / "oracle.json"
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    invariant_checkpoint = tmp_path / "invariant.safetensors"
    binding_checkpoint = tmp_path / "binding.safetensors"
    solve_output = tmp_path / "solve.json"
    benchmark_json = tmp_path / "benchmark.json"
    benchmark_csv = tmp_path / "benchmark.csv"

    assert (
        main(
            [
                "generate",
                "--node-count",
                "7",
                "--regime",
                "strongly_clustered",
                "--seed",
                "3",
                "--output",
                str(instance),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "oracle",
                str(instance),
                "--maximum-nodes",
                "10",
                "--output",
                str(oracle),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "collect",
                "--count",
                "5",
                "--node-counts",
                "7",
                "8",
                "--regimes",
                "clustered",
                "strongly_clustered",
                "--seed",
                "500",
                "--output",
                str(training),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "collect",
                "--count",
                "3",
                "--node-counts",
                "7",
                "8",
                "--regimes",
                "clustered",
                "strongly_clustered",
                "--seed",
                "600",
                "--output",
                str(validation),
            ]
        )
        == 0
    )
    common = [
        "--validation",
        str(validation),
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--hidden-dim",
        "8",
        "--hidden-layers",
        "1",
        "--negative-ratio",
        "3",
        "--minimum-negatives",
        "8",
        "--validation-budget",
        "3",
        "--patience",
        "1",
    ]
    assert (
        main(
            [
                "train",
                str(training),
                *common,
                "--target",
                "invariant",
                "--checkpoint",
                str(invariant_checkpoint),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                str(training),
                *common,
                "--target",
                "binding",
                "--checkpoint",
                str(binding_checkpoint),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "solve",
                str(instance),
                "--mode",
                "learned",
                "--checkpoint",
                str(invariant_checkpoint),
                "--budget",
                "3",
                "--output",
                str(solve_output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "benchmark",
                str(validation),
                "--invariant-checkpoint",
                str(invariant_checkpoint),
                "--binding-checkpoint",
                str(binding_checkpoint),
                "--budget",
                "3",
                "--bootstrap-draws",
                "5",
                "--output-json",
                str(benchmark_json),
                "--output-csv",
                str(benchmark_csv),
            ]
        )
        == 0
    )

    assert json.loads(oracle.read_text(encoding="utf-8"))["cold_constraint_generation"]["certified"]
    assert json.loads(solve_output.read_text(encoding="utf-8"))["result"]["certified"]
    benchmark = json.loads(benchmark_json.read_text(encoding="utf-8"))
    assert benchmark["metadata"]["all_results_certified"]
    assert benchmark_csv.exists()


def test_cli_returns_structured_error(tmp_path: Path, capsys: object) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["solve", str(bad)]) == 2
