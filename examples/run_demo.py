"""Run a tiny exact comparison without persisting a training corpus."""

from warmcg.dataset import label_record
from warmcg.domain import generate_instance
from warmcg.solver import run_constraint_generation
from warmcg.warmstart import select_warm_start


def main() -> None:
    instance = generate_instance(
        node_count=9,
        regime="strongly_clustered",
        seed=42,
    )
    record = label_record(instance)
    cold = run_constraint_generation(instance)
    oracle_selection = select_warm_start(
        instance,
        method="oracle_invariant_full",
        budget=0,
        record=record,
    )
    warm = run_constraint_generation(instance, initial_cuts=oracle_selection.cuts)
    print(
        {
            "exact_objective": record.exact_tour.objective,
            "cold_master_solves": cold.master_solve_count,
            "one_shot_core_size": len(oracle_selection.cuts),
            "warm_master_solves": warm.master_solve_count,
            "both_verified": cold.held_karp_verified and warm.held_karp_verified,
        }
    )


if __name__ == "__main__":
    main()
