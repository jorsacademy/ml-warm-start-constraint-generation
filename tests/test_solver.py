from __future__ import annotations

import math

import pytest

from warmcg.domain import canonical_cut, generate_instance, solve_held_karp
from warmcg.solver import (
    build_one_shot_core,
    run_constraint_generation,
    solve_master,
    violated_component_cuts,
)


def test_degree_master_is_two_regular_and_may_be_disconnected() -> None:
    instance = generate_instance(node_count=8, regime="strongly_clustered", seed=0)
    master = solve_master(instance)
    assert len(master.selected_edges) == instance.node_count
    assert all(len(component) >= 3 for component in master.components)
    if not master.connected:
        cuts = violated_component_cuts(instance, master)
        assert cuts
        assert len({cut.nodes for cut in cuts}) == len(cuts)


def test_exact_constraint_generation_agrees_with_held_karp() -> None:
    instance = generate_instance(node_count=9, regime="clustered", seed=12)
    result = run_constraint_generation(instance)
    oracle = solve_held_karp(instance)
    assert result.certified
    assert result.held_karp_verified
    assert math.isclose(result.solution.objective, oracle.objective, abs_tol=1e-9)
    assert result.master_solve_count >= 1
    assert result.iterations[-1].component_count == 1


def test_valid_preloaded_cuts_cannot_change_the_optimum() -> None:
    instance = generate_instance(node_count=8, regime="uniform", seed=4)
    cuts = (
        canonical_cut(instance.node_count, (1, 2, 3)),
        canonical_cut(instance.node_count, (4, 5)),
    )
    result = run_constraint_generation(instance, initial_cuts=cuts)
    oracle = solve_held_karp(instance)
    assert math.isclose(result.solution.objective, oracle.objective, abs_tol=1e-9)
    assert set(cuts).issubset(set(result.active_cuts))


def test_greedy_core_recovers_one_shot_optimum_and_is_deletion_minimal() -> None:
    instance = generate_instance(node_count=8, regime="strongly_clustered", seed=3)
    cold = run_constraint_generation(instance)
    core = build_one_shot_core(
        instance,
        cold.generated_cuts,
        optimum_objective=cold.solution.objective,
    )
    warm = solve_master(instance, core.cuts)
    assert core.verified_one_shot
    assert core.deletion_minimal
    assert warm.connected
    assert math.isclose(warm.objective, cold.solution.objective, abs_tol=1e-9)
    for cut in core.cuts:
        trial = tuple(candidate for candidate in core.cuts if candidate != cut)
        reduced = solve_master(instance, trial)
        assert not (
            reduced.connected
            and math.isclose(reduced.objective, cold.solution.objective, abs_tol=1e-9)
        )


def test_constraint_generation_fails_closed_on_bad_iteration_limit() -> None:
    instance = generate_instance(node_count=8, regime="strongly_clustered", seed=0)
    with pytest.raises(RuntimeError):
        run_constraint_generation(instance, maximum_iterations=1)


def test_incompatible_cut_is_rejected() -> None:
    instance = generate_instance(node_count=8, regime="uniform", seed=1)
    other_cut = canonical_cut(9, (1, 2))
    with pytest.raises(ValueError):
        solve_master(instance, (other_cut,))
