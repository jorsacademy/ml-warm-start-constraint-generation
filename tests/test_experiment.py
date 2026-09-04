from __future__ import annotations

from pathlib import Path

from warmcg.experiment import ResearchConfig, ScenarioConfig, run_research


def test_research_config_round_trip_and_tiny_protocol(tmp_path: Path) -> None:
    config = ResearchConfig(
        training_count=6,
        training_node_counts=(7, 8),
        training_regimes=("clustered", "strongly_clustered"),
        training_seed=700,
        validation_count=3,
        validation_node_counts=(7, 8),
        validation_regimes=("clustered", "strongly_clustered"),
        validation_seed=800,
        hidden_dim=8,
        hidden_layers=1,
        epochs=1,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=1e-5,
        negative_ratio=3,
        minimum_negatives_per_instance=8,
        patience=1,
        warm_start_budget=3,
        bootstrap_draws=5,
        bootstrap_seed=900,
        model_seed=1000,
        scenarios=(
            ScenarioConfig(
                name="interpolation",
                count=2,
                node_count=7,
                regime="strongly_clustered",
                seed=1100,
            ),
        ),
    )
    restored = ResearchConfig.from_dict(config.to_dict())
    assert restored == config
    report = run_research(restored, checkpoint_directory=tmp_path / "checkpoints")
    assert report.config_fingerprint
    assert len(report.evaluations) == 1
    assert report.evaluations[0].metadata["all_results_certified"] is True
    assert Path(report.checkpoints["invariant"]).exists()
    assert Path(report.checkpoints["binding"]).exists()
