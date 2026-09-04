"""Permutation-invariant geometric features for candidate subtour constraints."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from warmcg.domain import CutConstraint, TSPInstance

FEATURE_SCHEMA_VERSION = "tsp-sec-geometry-v1"
FEATURE_NAMES = (
    "smaller_side_fraction",
    "cut_edge_fraction",
    "size_balance",
    "cut_mean_over_global",
    "cut_std_over_global",
    "cut_min_over_global",
    "cut_max_over_global",
    "lower_within_mean_over_global",
    "upper_within_mean_over_global",
    "lower_within_std_over_global",
    "upper_within_std_over_global",
    "cut_to_within_mean_ratio",
    "centroid_separation_over_global",
    "lower_radius_over_global",
    "upper_radius_over_global",
    "nearest_1_cross_fraction",
    "nearest_2_cross_fraction",
    "nearest_3_cross_fraction",
    "cut_cost_fraction",
    "node_count_over_20",
    "global_distance_coefficient_of_variation",
)


@dataclass(frozen=True, slots=True)
class CandidateFeatureBatch:
    cuts: tuple[CutConstraint, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.values.shape != (len(self.cuts), len(FEATURE_NAMES)):
            raise ValueError("candidate feature matrix has an incompatible shape")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("candidate feature matrix contains non-finite values")


def _edge_arrays(instance: TSPInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    distances: list[float] = []
    for first in range(instance.node_count):
        for second in range(first + 1, instance.node_count):
            left.append(first)
            right.append(second)
            distances.append(float(instance.distance_matrix[first, second]))
    return (
        np.asarray(left, dtype=np.int16),
        np.asarray(right, dtype=np.int16),
        np.asarray(distances, dtype=np.float64),
    )


def _masked_statistics(
    mask: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sum(mask, axis=1, dtype=np.int64)
    if np.any(counts <= 0):
        raise RuntimeError("feature mask contains an empty row")
    totals = mask @ values
    means = totals / counts
    squared_totals = mask @ (values * values)
    variances = np.maximum(0.0, squared_totals / counts - means * means)
    standard_deviations = np.sqrt(variances)
    minimums = np.min(np.where(mask, values[None, :], np.inf), axis=1)
    maximums = np.max(np.where(mask, values[None, :], -np.inf), axis=1)
    return means, standard_deviations, minimums, maximums


def candidate_features(
    instance: TSPInstance,
    cuts: Sequence[CutConstraint],
    *,
    chunk_size: int = 4096,
) -> CandidateFeatureBatch:
    """Compute complement-invariant features from geometry only, before any MILP solve."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    ordered_cuts = tuple(cuts)
    if any(cut.node_count != instance.node_count for cut in ordered_cuts):
        raise ValueError("candidate cut node count does not match the instance")
    if len({cut.nodes for cut in ordered_cuts}) != len(ordered_cuts):
        raise ValueError("candidate cuts must be unique")
    if not ordered_cuts:
        return CandidateFeatureBatch(
            cuts=(),
            values=np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
        )

    n = instance.node_count
    coordinates = instance.coordinate_array
    left, right, edge_distances = _edge_arrays(instance)
    global_mean = float(np.mean(edge_distances))
    global_std = float(np.std(edge_distances))
    global_total = float(np.sum(edge_distances))
    scale = max(global_mean, 1e-12)
    nearest = np.argsort(instance.distance_matrix + np.eye(n) * 1e9, axis=1)[:, :3]
    coordinate_norm_squared = np.sum(coordinates * coordinates, axis=1)
    result = np.empty((len(ordered_cuts), len(FEATURE_NAMES)), dtype=np.float64)

    for start in range(0, len(ordered_cuts), chunk_size):
        stop = min(len(ordered_cuts), start + chunk_size)
        chunk = ordered_cuts[start:stop]
        membership = np.zeros((len(chunk), n), dtype=bool)
        for row, cut in enumerate(chunk):
            membership[row, np.asarray(cut.nodes, dtype=np.int16)] = True
        member_float = membership.astype(np.float64)
        complement_float = 1.0 - member_float
        sizes = np.sum(member_float, axis=1)
        complement_sizes = n - sizes

        member_left = membership[:, left]
        member_right = membership[:, right]
        crossing = np.logical_xor(member_left, member_right)
        internal_a = np.logical_and(member_left, member_right)
        internal_b = np.logical_and(~member_left, ~member_right)

        cut_mean, cut_std, cut_min, cut_max = _masked_statistics(crossing, edge_distances)
        within_a_mean, within_a_std, _, _ = _masked_statistics(internal_a, edge_distances)
        within_b_mean, within_b_std, _, _ = _masked_statistics(internal_b, edge_distances)
        lower_within_mean = np.minimum(within_a_mean, within_b_mean)
        upper_within_mean = np.maximum(within_a_mean, within_b_mean)
        lower_within_std = np.minimum(within_a_std, within_b_std)
        upper_within_std = np.maximum(within_a_std, within_b_std)

        centroid_a = (member_float @ coordinates) / sizes[:, None]
        centroid_b = (complement_float @ coordinates) / complement_sizes[:, None]
        centroid_separation = np.sqrt(np.sum((centroid_a - centroid_b) ** 2, axis=1))
        mean_norm_a = (member_float @ coordinate_norm_squared) / sizes
        mean_norm_b = (complement_float @ coordinate_norm_squared) / complement_sizes
        radius_a = np.sqrt(np.maximum(0.0, mean_norm_a - np.sum(centroid_a**2, axis=1)))
        radius_b = np.sqrt(np.maximum(0.0, mean_norm_b - np.sum(centroid_b**2, axis=1)))
        lower_radius = np.minimum(radius_a, radius_b)
        upper_radius = np.maximum(radius_a, radius_b)

        nearest_cross: list[np.ndarray] = []
        row_indices = np.arange(len(chunk))[:, None]
        for rank in range(3):
            neighbor_membership = membership[row_indices, nearest[None, :, rank]]
            nearest_cross.append(np.mean(membership != neighbor_membership, axis=1))

        cut_counts = sizes * complement_sizes
        cut_totals = crossing @ edge_distances
        within_reference = 0.5 * (within_a_mean + within_b_mean)
        features = np.column_stack(
            (
                np.minimum(sizes, complement_sizes) / n,
                cut_counts / instance.edge_count,
                4.0 * sizes * complement_sizes / (n * n),
                cut_mean / scale,
                cut_std / scale,
                cut_min / scale,
                cut_max / scale,
                lower_within_mean / scale,
                upper_within_mean / scale,
                lower_within_std / scale,
                upper_within_std / scale,
                cut_mean / np.maximum(within_reference, 1e-12),
                centroid_separation / scale,
                lower_radius / scale,
                upper_radius / scale,
                nearest_cross[0],
                nearest_cross[1],
                nearest_cross[2],
                cut_totals / max(global_total, 1e-12),
                np.full(len(chunk), n / 20.0, dtype=np.float64),
                np.full(
                    len(chunk),
                    global_std / scale,
                    dtype=np.float64,
                ),
            )
        )
        if not np.all(np.isfinite(features)):
            raise RuntimeError("candidate feature extraction produced non-finite values")
        result[start:stop] = features

    return CandidateFeatureBatch(cuts=ordered_cuts, values=result.astype(np.float32))


def compactness_heuristic_scores(features: np.ndarray) -> np.ndarray:
    """Score geometrically separated, internally compact cuts without solver information."""

    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("heuristic feature matrix has an incompatible shape")
    score = (
        1.25 * features[:, FEATURE_NAMES.index("cut_to_within_mean_ratio")]
        + 0.65 * features[:, FEATURE_NAMES.index("centroid_separation_over_global")]
        - 0.85 * features[:, FEATURE_NAMES.index("nearest_1_cross_fraction")]
        - 0.35 * features[:, FEATURE_NAMES.index("nearest_2_cross_fraction")]
        - 0.10 * features[:, FEATURE_NAMES.index("cut_edge_fraction")]
    )
    if not np.all(np.isfinite(score)):
        raise RuntimeError("heuristic scorer produced non-finite values")
    return score.astype(np.float64)


def rank_top_k(
    cuts: Sequence[CutConstraint],
    scores: Sequence[float],
    budget: int,
) -> tuple[CutConstraint, ...]:
    """Select a deterministic score-ranked subset under a fixed preload budget."""

    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if len(cuts) != len(scores):
        raise ValueError("cuts and scores must be aligned")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    order = sorted(
        range(len(cuts)),
        key=lambda index: (
            float(scores[index]),
            -cuts[index].size,
            tuple(-node for node in cuts[index].nodes),
        ),
        reverse=True,
    )
    return tuple(cuts[index] for index in order[: min(budget, len(order))])
