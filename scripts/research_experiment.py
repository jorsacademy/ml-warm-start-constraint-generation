"""Execute the checked-in frozen research protocol."""

from pathlib import Path

from warmcg.experiment import ResearchConfig, run_research
from warmcg.utils import as_object_dict, read_json, write_json


def main() -> None:
    config_path = Path("configs/research_v1.json")
    config = ResearchConfig.from_dict(
        as_object_dict(read_json(config_path), name="research configuration")
    )
    report = run_research(config, checkpoint_directory="artifacts/checkpoints")
    write_json(report.to_dict(), "artifacts/research-report.json")


if __name__ == "__main__":
    main()
