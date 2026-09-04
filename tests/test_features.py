from __future__ import annotations

import numpy as np
import pytest

from warmcg.domain import canonical_cut, enumerate_candidate_cuts, generate_instance
from warmcg.features import (
    FEATURE_NAMES,
    candidate_features,
    compactness_heuristic_scores,
    rank_top_k,
)


def test_candidate_features_are_finite_and_complement_invariant() -> None:
    instance = generate_instance(node_count=8, regime="clustered", seed=8)
    cut = canonical_cut(8, (1, 2, 3))
    complement = canonical_cut(8, (0, 4, 5, 6, 7))
    first = candidate_features(instance, (cut,)).values
    second = candidate_features(instance, (complement,)).values
    assert first.shape == (1, len(FEATURE_NAMES))
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second)


def test_feature_extraction_rejects_duplicates() -> None:
    instance = generate_instance(node_count=7, seed=1)
    cut = canonical_cut(7, (1, 2))
    with pytest.raises(ValueError):
        candidate_features(instance, (cut, cut))


def test_compactness_scoring_and_top_k_are_deterministic() -> None:
    instance = generate_instance(node_count=9, regime="strongly_clustered", seed=2)
    cuts = enumerate_candidate_cuts(instance.node_count)
    features = candidate_features(instance, cuts).values
    scores = compactness_heuristic_scores(features)
    first = rank_top_k(cuts, scores, 7)
    second = rank_top_k(cuts, scores, 7)
    assert first == second
    assert len(first) == 7
    assert len(set(first)) == 7


def test_rank_top_k_rejects_nonfinite_scores() -> None:
    cuts = enumerate_candidate_cuts(6)
    scores = np.zeros(len(cuts), dtype=float)
    scores[0] = np.nan
    with pytest.raises(ValueError):
        rank_top_k(cuts, scores, 2)
