"""Command-line workflows for data collection, training, and exact evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from warmcg.dataset import (
    LabelTarget,
    generate_dataset,
    label_record,
    load_dataset,
    save_dataset,
)
from warmcg.domain import (
    SUPPORTED_REGIMES,
    TSPInstance,
    generate_instance,
    solve_brute_force,
    solve_held_karp,
)
from warmcg.evaluation import evaluate_dataset, save_report_csv
from warmcg.experiment import ResearchConfig, run_research
from warmcg.model import (
    ConstraintScorerConfig,
    load_checkpoint,
    save_checkpoint,
)
from warmcg.solver import run_constraint_generation
from warmcg.training import TrainingConfig, train_constraint_scorer
from warmcg.utils import as_object_dict, read_json, write_json
from warmcg.warmstart import select_warm_start


def _instance_from_file(path: str | Path) -> TSPInstance:
    payload = as_object_dict(read_json(path), name="instance file")
    return TSPInstance.from_dict(payload)


def _write_or_print(payload: object, output: str | None) -> None:
    if output is None:
        print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    else:
        write_json(payload, output)


def _command_generate(args: argparse.Namespace) -> dict[str, object]:
    instance = generate_instance(
        node_count=args.node_count,
        regime=args.regime,
        seed=args.seed,
    )
    payload = instance.to_dict()
    _write_or_print(payload, args.output)
    return payload


def _command_oracle(args: argparse.Namespace) -> dict[str, object]:
    instance = _instance_from_file(args.input)
    record = label_record(instance, held_karp_maximum_nodes=args.maximum_nodes)
    cold = run_constraint_generation(
        instance,
        verify_with_held_karp=True,
        held_karp_maximum_nodes=args.maximum_nodes,
    )
    held_karp = solve_held_karp(instance, maximum_nodes=args.maximum_nodes)
    brute_force = (
        solve_brute_force(instance, maximum_nodes=args.brute_force_maximum_nodes)
        if instance.node_count <= args.brute_force_maximum_nodes
        else None
    )
    payload = {
        "instance": instance.to_dict(),
        "cold_constraint_generation": cold.to_dict(),
        "held_karp": held_karp.to_dict(),
        "brute_force": None if brute_force is None else brute_force.to_dict(),
        "labels": record.to_dict(),
    }
    _write_or_print(payload, args.output)
    return payload


def _command_collect(args: argparse.Namespace) -> dict[str, object]:
    dataset = generate_dataset(
        count=args.count,
        node_counts=tuple(args.node_counts),
        regimes=tuple(args.regimes),
        seed=args.seed,
        held_karp_maximum_nodes=args.maximum_nodes,
    )
    save_dataset(dataset, args.output)
    payload = dataset.to_metadata()
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return payload


def _command_train(args: argparse.Namespace) -> dict[str, object]:
    training = load_dataset(args.training)
    validation = load_dataset(args.validation)
    target = cast(LabelTarget, args.target)
    model, report = train_constraint_scorer(
        training,
        validation,
        target=target,
        model_config=ConstraintScorerConfig(
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
        ),
        training_config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            negative_ratio=args.negative_ratio,
            minimum_negatives_per_instance=args.minimum_negatives,
            validation_budget=args.validation_budget,
            patience=args.patience,
            gradient_clip_norm=args.gradient_clip_norm,
            seed=args.seed,
        ),
    )
    payload = report.to_dict()
    save_checkpoint(
        model,
        args.checkpoint,
        target=target,
        metadata={"training_report": payload},
    )
    if args.output_report is not None:
        write_json(payload, args.output_report)
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return payload


def _command_solve(args: argparse.Namespace) -> dict[str, object]:
    instance = _instance_from_file(args.input)
    model = None
    target: LabelTarget | None = None
    method = args.mode
    if args.mode == "learned":
        if args.checkpoint is None:
            raise ValueError("learned solve mode requires --checkpoint")
        model, target, _ = load_checkpoint(args.checkpoint)
        if target == "binding":
            method = "binding_model"
        elif target == "invariant":
            method = "invariant_model"
        else:
            raise ValueError("solve currently accepts invariant or binding checkpoints")
    selection = select_warm_start(
        instance,
        method=method,
        budget=args.budget,
        seed=args.seed,
        model=model,
    )
    result = run_constraint_generation(
        instance,
        initial_cuts=selection.cuts,
        verify_with_held_karp=instance.node_count <= args.maximum_nodes,
        held_karp_maximum_nodes=args.maximum_nodes,
    )
    payload = {
        "instance": instance.to_dict(),
        "checkpoint_target": target,
        "warm_start": selection.to_dict(),
        "result": result.to_dict(),
    }
    _write_or_print(payload, args.output)
    return payload


def _command_benchmark(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_dataset(args.dataset)
    invariant_model, invariant_target, invariant_metadata = load_checkpoint(
        args.invariant_checkpoint
    )
    binding_model, binding_target, binding_metadata = load_checkpoint(args.binding_checkpoint)
    if invariant_target != "invariant":
        raise ValueError("--invariant-checkpoint does not contain an invariant-target model")
    if binding_target != "binding":
        raise ValueError("--binding-checkpoint does not contain a binding-target model")
    report = evaluate_dataset(
        dataset,
        scenario=args.scenario,
        budget=args.budget,
        invariant_model=invariant_model,
        binding_model=binding_model,
        random_seed=args.random_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_draws=args.bootstrap_draws,
    )
    payload = report.to_dict()
    payload["checkpoint_metadata"] = {
        "invariant": invariant_metadata,
        "binding": binding_metadata,
    }
    write_json(payload, args.output_json)
    save_report_csv(report, args.output_csv)
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return payload


def _command_research(args: argparse.Namespace) -> dict[str, object]:
    raw = as_object_dict(read_json(args.config), name="research configuration")
    config = ResearchConfig.from_dict(raw)
    report = run_research(config, checkpoint_directory=args.checkpoint_directory)
    payload = report.to_dict()
    write_json(payload, args.output_report)
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warmcg",
        description="Verification-first ML warm starts for exact TSP constraint generation.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate one Euclidean TSP instance")
    generate.add_argument("--node-count", type=int, default=10)
    generate.add_argument("--regime", choices=SUPPORTED_REGIMES, default="uniform")
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output")
    generate.set_defaults(handler=_command_generate)

    oracle = subparsers.add_parser("oracle", help="run exact CG and independent tour oracles")
    oracle.add_argument("input")
    oracle.add_argument("--maximum-nodes", type=int, default=14)
    oracle.add_argument("--brute-force-maximum-nodes", type=int, default=9)
    oracle.add_argument("--output")
    oracle.set_defaults(handler=_command_oracle)

    collect = subparsers.add_parser("collect", help="build an exact-labeled JSONL corpus")
    collect.add_argument("--count", type=int, default=48)
    collect.add_argument("--node-counts", type=int, nargs="+", default=[8, 10, 12])
    collect.add_argument("--regimes", nargs="+", choices=SUPPORTED_REGIMES, default=["uniform"])
    collect.add_argument("--seed", type=int, default=1000)
    collect.add_argument("--maximum-nodes", type=int, default=14)
    collect.add_argument("--output", required=True)
    collect.set_defaults(handler=_command_collect)

    train = subparsers.add_parser("train", help="train one constraint-set target scorer")
    train.add_argument("training")
    train.add_argument("--validation", required=True)
    train.add_argument(
        "--target",
        choices=("invariant", "trajectory", "binding", "initial"),
        default="invariant",
    )
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--negative-ratio", type=int, default=12)
    train.add_argument("--minimum-negatives", type=int, default=48)
    train.add_argument("--validation-budget", type=int, default=8)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--gradient-clip-norm", type=float, default=5.0)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--hidden-layers", type=int, default=2)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--output-report")
    train.set_defaults(handler=_command_train)

    solve = subparsers.add_parser("solve", help="solve one instance with an exact CG loop")
    solve.add_argument("input")
    solve.add_argument(
        "--mode",
        choices=("cold", "random", "compactness", "learned"),
        default="cold",
    )
    solve.add_argument("--checkpoint")
    solve.add_argument("--budget", type=int, default=8)
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--maximum-nodes", type=int, default=14)
    solve.add_argument("--output")
    solve.set_defaults(handler=_command_solve)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare matched-budget warm starts on an exact-labeled corpus"
    )
    benchmark.add_argument("dataset")
    benchmark.add_argument("--invariant-checkpoint", required=True)
    benchmark.add_argument("--binding-checkpoint", required=True)
    benchmark.add_argument("--scenario", default="benchmark")
    benchmark.add_argument("--budget", type=int, default=8)
    benchmark.add_argument("--random-seed", type=int, default=0)
    benchmark.add_argument("--bootstrap-seed", type=int, default=0)
    benchmark.add_argument("--bootstrap-draws", type=int, default=500)
    benchmark.add_argument("--output-json", required=True)
    benchmark.add_argument("--output-csv", required=True)
    benchmark.set_defaults(handler=_command_benchmark)

    research = subparsers.add_parser("research", help="run the frozen train/evaluate protocol")
    research.add_argument("--config", default="configs/research_v1.json")
    research.add_argument("--checkpoint-directory", required=True)
    research.add_argument("--output-report", required=True)
    research.set_defaults(handler=_command_research)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        handler = cast(Callable[[argparse.Namespace], object], args.handler)
        handler(args)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": str(exc)},
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
