#!/usr/bin/env python
"""Generate the figures.

Usage:
    python -m figures.generate_all                       # all figures
    python -m figures.generate_all overview drift        # selected figures
    python -m figures.generate_all --list                # show available figures
"""
import importlib
import sys

FIGURES = {
    # Figures used in the paper.
    "overview": ("fig_overview", "Conceptual overview (paper Figure 1)"),
    "mechanism": ("fig_merge_mechanism", "Where the default merge loses information (paper Figure 2)"),
    "drift": ("fig_predicate_drift", "Rule counts and predicate midpoints across replay seeds (paper Figure 3)"),
    # Supporting figures.
    "noise": ("fig_noise_severity", "Fidelity and agreement under increasing noise"),
    "decoupling": ("fig_fidelity_vs_stability", "Fidelity vs prediction-agreement scatter"),
    "pareto": ("fig_pareto_frontier", "Near-match overlap vs fidelity Pareto frontier"),
    "consensus-ablation": ("fig_consensus_ablation", "Ensemble size x support threshold ablation"),
    "env-perturbation": ("fig_env_perturbation", "Return vs agreement under environment perturbation"),
    "local-consistency": ("fig_local_consistency", "Local explanation consistency curves"),
    "minigrid": ("fig_minigrid", "MiniGrid main comparison"),
    "complexity": ("fig_complexity", "Complexity extrapolation across environments"),
    "policy-family-cp": ("fig_policy_family_cartpole", "PPO vs DQN on CartPole"),
    "policy-family-ll": ("fig_policy_family_lunarlander", "PPO vs DQN on LunarLander"),
    "ppo-vs-dqn": ("fig_ppo_vs_dqn", "Cross-environment PPO vs DQN summary"),
    "suite-summary": ("fig_suite_summary", "Compact summary of the robustness suite"),
    "soft-support-ablation": ("fig_soft_support_ablation", "SoftSupport ablation heatmap"),
    "soft-support-pareto": ("fig_soft_support_pareto", "SoftSupport Pareto frontier"),
    "tree-structure": ("fig_tree_structure", "Decision-tree structural diagnostic (3 outputs)"),
    "merge-stages": ("fig_merge_stage_decomposition", "Merge failure-stage decomposition"),
    "merge-stages-by-env": ("fig_merge_stage_by_env", "Environment-specific failure modes"),
    "distortion": ("fig_geometric_distortion", "Geometric distortion diagnosis"),
    "monotonicity": ("fig_condition_monotonicity", "Condition monotonicity analysis"),
    "boundary": ("fig_boundary_crossing", "Boundary-crossing summary"),
    "boundary-case": ("fig_boundary_case_study", "Boundary-crossing case study"),
    "minigrid-transfer": ("fig_minigrid_transfer", "External validity on MiniGrid + PPO"),
    "weighted-vote": ("fig_weighted_vote", "Weighted RuleVote comparison"),
    "merge-repairs": ("fig_merge_repairs", "Merge-repair study (6 outputs)"),
}


def main():
    if "--list" in sys.argv:
        print("Available figures:")
        for key, (module_name, description) in FIGURES.items():
            print(f"  {key}: {description}  ({module_name}.py)")
        return

    requested = [arg.lower() for arg in sys.argv[1:] if not arg.startswith("--")]
    if not requested:
        requested = list(FIGURES.keys())

    print(f"Generating {len(requested)} figure(s)...\n")
    for key in requested:
        if key not in FIGURES:
            print(f"  Unknown figure: {key}")
            continue
        module_name, description = FIGURES[key]
        print(f"[{key}] {description}")
        try:
            module = importlib.import_module(f"figures.{module_name}")
            module.main()
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
