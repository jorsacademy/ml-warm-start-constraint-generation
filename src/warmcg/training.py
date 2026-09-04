"""Instance-grouped training and ranking diagnostics for constraint warm starts."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn

from warmcg.dataset import LabelTarget, WarmStartDataset, target_label_vector
from warmcg.domain import enumerate_candidate_cuts
from warmcg.features import candidate_features, compactness_heuristic_scores
from warmcg.model import ConstraintScorer, ConstraintScorerConfig, score_feature_matrix


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    negative_ratio: int = 12
    minimum_negatives_per_instance: int = 48
    validation_budget: int = 8
    patience: int = 10
    gradient_clip_norm: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.negative_ratio <= 0 or self.minimum_negatives_per_instance <= 0:
            raise ValueError("negative-sampling parameters must be positive")
        if self.validation_budget <= 0 or self.patience <= 0:
            raise ValueError("validation budget and patience must be positive")
        if self.gradient_clip_norm <= 0.0 or self.seed < 0:
            raise ValueError("gradient clipping and seed values are invalid")


@dataclass(frozen=True, slots=True)
class PreparedSamples:
    features: np.ndarray
    labels: np.ndarray
    group_ids: np.ndarray
    instance_count: int
    full_candidate_count: int

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("sample features must be a matrix")
        if self.labels.shape != (self.features.shape[0],):
            raise ValueError("sample labels are not aligned")
        if self.group_ids.shape != self.labels.shape:
            raise ValueError("sample group identifiers are not aligned")
        if not np.all(np.isfinite(self.features)) or not np.all(np.isfinite(self.labels)):
            raise ValueError("prepared samples contain non-finite values")
        if self.features.shape[0] == 0:
            raise ValueError("prepared sample set is empty")

    @property
    def positive_count(self) -> int:
        return int(np.sum(self.labels > 0.5))

    @property
    def negative_count(self) -> int:
        return int(self.labels.size - self.positive_count)

    @property
    def positive_rate(self) -> float:
        return self.positive_count / self.labels.size


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    instance_count: int
    candidate_count: int
    positive_count: int
    average_precision: float
    top_k_precision: float
    top_k_recall: float
    any_positive_hit_rate: float
    mean_positive_score: float | None
    mean_negative_score: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochRecord:
    epoch: int
    training_loss: float
    validation_loss: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingReport:
    target: LabelTarget
    config: dict[str, object]
    model_config: dict[str, object]
    best_epoch: int
    history: tuple[EpochRecord, ...]
    training_sample_count: int
    validation_sample_count: int
    training_positive_rate: float
    validation_positive_rate: float
    training_dataset_fingerprint: str
    validation_dataset_fingerprint: str
    training_ranking: RankingMetrics
    validation_ranking: RankingMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "config": self.config,
            "model_config": self.model_config,
            "best_epoch": self.best_epoch,
            "history": [row.to_dict() for row in self.history],
            "training_sample_count": self.training_sample_count,
            "validation_sample_count": self.validation_sample_count,
            "training_positive_rate": self.training_positive_rate,
            "validation_positive_rate": self.validation_positive_rate,
            "training_dataset_fingerprint": self.training_dataset_fingerprint,
            "validation_dataset_fingerprint": self.validation_dataset_fingerprint,
            "training_ranking": self.training_ranking.to_dict(),
            "validation_ranking": self.validation_ranking.to_dict(),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_samples(
    dataset: WarmStartDataset,
    *,
    target: LabelTarget,
    negative_ratio: int,
    minimum_negatives_per_instance: int,
    seed: int,
) -> PreparedSamples:
    """Keep every positive plus deterministic hard/random negatives per whole instance."""

    if negative_ratio <= 0 or minimum_negatives_per_instance <= 0:
        raise ValueError("negative sampling parameters must be positive")
    rng = np.random.default_rng(seed)
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    group_rows: list[np.ndarray] = []
    full_count = 0

    for group_id, record in enumerate(dataset.records):
        candidates = enumerate_candidate_cuts(record.instance.node_count)
        batch = candidate_features(record.instance, candidates)
        labels = target_label_vector(record, candidates, target)
        full_count += len(candidates)
        positive_indices = np.flatnonzero(labels > 0.5)
        negative_indices = np.flatnonzero(labels <= 0.5)
        desired_negatives = max(
            minimum_negatives_per_instance,
            negative_ratio * max(1, positive_indices.size),
        )
        desired_negatives = min(desired_negatives, negative_indices.size)
        heuristic_scores = compactness_heuristic_scores(batch.values)
        hard_count = min(desired_negatives // 2, negative_indices.size)
        hard_order = negative_indices[
            np.argsort(heuristic_scores[negative_indices], kind="mergesort")[::-1]
        ]
        selected_hard = hard_order[:hard_count]
        remaining = np.setdiff1d(negative_indices, selected_hard, assume_unique=True)
        random_count = desired_negatives - selected_hard.size
        selected_random = (
            rng.choice(remaining, size=random_count, replace=False)
            if random_count > 0
            else np.empty(0, dtype=np.int64)
        )
        selected = np.unique(np.concatenate((positive_indices, selected_hard, selected_random)))
        if selected.size == 0:
            raise RuntimeError("negative sampling selected no candidate rows")
        feature_rows.append(batch.values[selected])
        label_rows.append(labels[selected])
        group_rows.append(np.full(selected.size, group_id, dtype=np.int32))

    return PreparedSamples(
        features=np.concatenate(feature_rows, axis=0).astype(np.float32),
        labels=np.concatenate(label_rows, axis=0).astype(np.float32),
        group_ids=np.concatenate(group_rows, axis=0),
        instance_count=len(dataset.records),
        full_candidate_count=full_count,
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels > 0.5))
    if positives == 0:
        return 1.0
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered_labels = labels[order] > 0.5
    cumulative = np.cumsum(ordered_labels)
    precision = cumulative / (np.arange(labels.size) + 1)
    return float(np.sum(precision * ordered_labels) / positives)


def evaluate_ranking(
    model: ConstraintScorer,
    dataset: WarmStartDataset,
    *,
    target: LabelTarget,
    budget: int,
) -> RankingMetrics:
    """Evaluate the full candidate universe on each held-out instance."""

    if budget <= 0:
        raise ValueError("budget must be positive")
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    top_precisions: list[float] = []
    top_recalls: list[float] = []
    hit_rates: list[float] = []
    candidate_count = 0

    for record in dataset.records:
        candidates = enumerate_candidate_cuts(record.instance.node_count)
        features = candidate_features(record.instance, candidates).values
        labels = target_label_vector(record, candidates, target)
        scores = score_feature_matrix(model, features)
        candidate_count += len(candidates)
        all_labels.append(labels)
        all_scores.append(scores)
        order = np.argsort(scores, kind="mergesort")[::-1]
        selected = order[: min(budget, order.size)]
        true_count = int(np.sum(labels > 0.5))
        hits = int(np.sum(labels[selected] > 0.5))
        top_precisions.append(hits / max(1, selected.size))
        top_recalls.append(hits / max(1, true_count))
        hit_rates.append(float(hits > 0) if true_count > 0 else 1.0)

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    positive_scores = scores[labels > 0.5]
    negative_scores = scores[labels <= 0.5]
    return RankingMetrics(
        instance_count=len(dataset.records),
        candidate_count=candidate_count,
        positive_count=int(np.sum(labels > 0.5)),
        average_precision=_average_precision(labels, scores),
        top_k_precision=float(np.mean(np.asarray(top_precisions, dtype=float))),
        top_k_recall=float(np.mean(np.asarray(top_recalls, dtype=float))),
        any_positive_hit_rate=float(np.mean(np.asarray(hit_rates, dtype=float))),
        mean_positive_score=(float(np.mean(positive_scores)) if positive_scores.size else None),
        mean_negative_score=(float(np.mean(negative_scores)) if negative_scores.size else None),
    )


def _loss_on_samples(
    model: ConstraintScorer,
    samples: PreparedSamples,
    criterion: nn.BCEWithLogitsLoss,
    *,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, samples.labels.size, batch_size):
            stop = min(samples.labels.size, start + batch_size)
            features = torch.as_tensor(
                samples.features[start:stop], dtype=torch.float32, device=model.device
            )
            labels = torch.as_tensor(
                samples.labels[start:stop], dtype=torch.float32, device=model.device
            )
            loss = criterion(model(features), labels)
            total += float(loss.detach().cpu()) * (stop - start)
            count += stop - start
    return total / max(1, count)


def train_constraint_scorer(
    training_dataset: WarmStartDataset,
    validation_dataset: WarmStartDataset,
    *,
    target: LabelTarget,
    model_config: ConstraintScorerConfig | None = None,
    training_config: TrainingConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[ConstraintScorer, TrainingReport]:
    """Train one target-specific scorer with early stopping on disjoint instances."""

    config = training_config or TrainingConfig()
    architecture = model_config or ConstraintScorerConfig()
    _seed_everything(config.seed)
    training_samples = prepare_samples(
        training_dataset,
        target=target,
        negative_ratio=config.negative_ratio,
        minimum_negatives_per_instance=config.minimum_negatives_per_instance,
        seed=config.seed,
    )
    validation_samples = prepare_samples(
        validation_dataset,
        target=target,
        negative_ratio=config.negative_ratio,
        minimum_negatives_per_instance=config.minimum_negatives_per_instance,
        seed=config.seed + 1,
    )
    if training_samples.positive_count == 0:
        raise ValueError(f"training corpus contains no positive {target!r} labels")

    mean = np.asarray(
        np.mean(training_samples.features, axis=0, dtype=np.float64),
        dtype=np.float32,
    )
    scale = np.asarray(
        np.std(training_samples.features, axis=0, dtype=np.float64),
        dtype=np.float32,
    )
    scale = np.asarray(np.where(scale < 1e-6, 1.0, scale), dtype=np.float32)
    model = ConstraintScorer(architecture).to(device)
    model.set_normalization(mean, scale)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    ratio = training_samples.negative_count / max(1, training_samples.positive_count)
    pos_weight = torch.tensor(min(30.0, max(1.0, ratio)), dtype=torch.float32, device=model.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = np.random.default_rng(config.seed + 2)
    history: list[EpochRecord] = []
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        order = generator.permutation(training_samples.labels.size)
        total = 0.0
        count = 0
        for start in range(0, order.size, config.batch_size):
            indices = order[start : start + config.batch_size]
            features = torch.as_tensor(
                training_samples.features[indices],
                dtype=torch.float32,
                device=model.device,
            )
            labels = torch.as_tensor(
                training_samples.labels[indices],
                dtype=torch.float32,
                device=model.device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), labels)
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("training gradient became non-finite")
            optimizer.step()
            batch_count = int(indices.size)
            total += float(loss.detach().cpu()) * batch_count
            count += batch_count
        training_loss = total / max(1, count)
        validation_loss = _loss_on_samples(
            model,
            validation_samples,
            criterion,
            batch_size=config.batch_size,
        )
        history.append(EpochRecord(epoch, training_loss, validation_loss))
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("training failed to produce a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    training_ranking = evaluate_ranking(
        model,
        training_dataset,
        target=target,
        budget=config.validation_budget,
    )
    validation_ranking = evaluate_ranking(
        model,
        validation_dataset,
        target=target,
        budget=config.validation_budget,
    )
    report = TrainingReport(
        target=target,
        config=asdict(config),
        model_config=asdict(architecture),
        best_epoch=best_epoch,
        history=tuple(history),
        training_sample_count=training_samples.labels.size,
        validation_sample_count=validation_samples.labels.size,
        training_positive_rate=training_samples.positive_rate,
        validation_positive_rate=validation_samples.positive_rate,
        training_dataset_fingerprint=training_dataset.fingerprint,
        validation_dataset_fingerprint=validation_dataset.fingerprint,
        training_ranking=training_ranking,
        validation_ranking=validation_ranking,
    )
    return model, report
