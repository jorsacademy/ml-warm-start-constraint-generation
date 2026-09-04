from __future__ import annotations

import math

import pytest

from warmcg.domain import (
    SUPPORTED_REGIMES,
    TSPInstance,
    candidate_cut_count,
    canonical_cut,
    cut_value_on_tour,
    enumerate_candidate_cuts,
    generate_instance,
    solve_brute_force,
    solve_held_karp,
)


def test_instance_generation_is_deterministic_across_regimes() -> None:
    for regime in SUPPORTED_REGIMES:
        first = generate_instance(node_count=8, regime=regime, seed=42)
        second = generate_instance(node_count=8, regime=regime, seed=42)
        assert first == second
        assert first.distance_matrix.shape == (8, 8)
        assert first.edge_count == 28


def test_instance_validation_rejects_duplicates_and_small_graphs() -> None:
    with pytest.raises(ValueError):
        TSPInstance(((0.0, 0.0),) * 5)
    with pytest.raises(ValueError):
        generate_instance(node_count=4)


def test_cut_canonicalization_identifies_complements() -> None:
    cut = canonical_cut(8, (0, 1, 2, 3))
    complement = canonical_cut(8, (4, 5, 6, 7))
    assert cut == complement
    assert 0 not in cut.nodes


def test_candidate_universe_has_one_representative_per_symmetric_cut() -> None:
    candidates = enumerate_candidate_cuts(8)
    assert len(candidates) == candidate_cut_count(8)
    assert len({candidate.nodes for candidate in candidates}) == len(candidates)
    assert all(0 not in candidate.nodes for candidate in candidates)


def test_held_karp_agrees_with_brute_force() -> None:
    instance = generate_instance(node_count=8, regime="uniform", seed=7)
    held_karp = solve_held_karp(instance)
    brute_force = solve_brute_force(instance)
    assert math.isclose(held_karp.objective, brute_force.objective, abs_tol=1e-10)
    assert set(held_karp.order) == set(range(instance.node_count))


def test_every_nontrivial_cut_is_valid_for_a_tour() -> None:
    instance = generate_instance(node_count=7, regime="ring", seed=3)
    tour = solve_held_karp(instance)
    values = [cut_value_on_tour(cut, tour) for cut in enumerate_candidate_cuts(7)]
    assert min(values) >= 2
    assert any(value == 2 for value in values)
