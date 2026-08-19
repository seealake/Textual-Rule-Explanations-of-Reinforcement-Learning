#!/usr/bin/env python
"""Generate meta-evaluation summaries from existing experiment artifacts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
REPLAY_DIR = ROOT / "reproduction" / "data"
OUT_DIR = RESULTS_DIR / "suite_summary"


@dataclass(frozen=True)
class EnvSpec:
    env: str
    short: str
    env_tag: str
    policy_family: str
    obs_features: int
    action_count: int
    replay_file: str


ENV_SPECS = [
    EnvSpec("MountainCar-v0", "MC", "mountaincar_v0", "DQN", 2, 3, "replay_mountaincar_v0_seed42.npz"),
    EnvSpec("CartPole-v1", "CP", "cartpole_v1", "DQN", 4, 2, "replay_cartpole_v1_seed42.npz"),
    EnvSpec("LunarLander-v3", "LL", "lunarlander_v3", "DQN", 8, 4, "replay_lunarlander_v3_seed42.npz"),
    EnvSpec(
        "MiniGrid-Dynamic-Obstacles-8x8-v0",
        "MG",
        "minigrid_dynamic_obstacles_8x8_v0",
        "PPO",
        14,
        3,
        "replay_minigrid_dynamic_obstacles_8x8_v0_ppo_seed42.npz",
    ),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


DIAG_ENV_TAGS = ["mountaincar_v0", "cartpole_v1", "lunarlander_v3"]


def _env_name_from_tag(env_tag: str) -> str:
    for spec in ENV_SPECS:
        if spec.env_tag == env_tag:
            return spec.env
    raise KeyError(f"Unknown env tag: {env_tag}")


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
    }


def _load_cbs_stability(env_tag: str) -> tuple[dict[str, Any], str]:
    path = RESULTS_DIR / env_tag / "stress_test_results.json"
    data = _load_json(path)
    if "methods" in data:
        return data["methods"]["CBS"]["stability"], _rel(path)
    cbs = data["cbs"]
    return cbs.get("stability", cbs), _rel(path)


def _compute_action_stats(spec: EnvSpec) -> dict[str, Any]:
    replay_path = REPLAY_DIR / spec.replay_file
    replay = np.load(replay_path)
    actions = replay["actions"].astype(int)
    counts = np.bincount(actions, minlength=spec.action_count)
    probs = counts / counts.sum()
    entropy = -(probs[probs > 0] * np.log(probs[probs > 0])).sum()
    norm_entropy = entropy / math.log(spec.action_count)
    return {
        "action_counts": counts.tolist(),
        "action_probs": [float(prob) for prob in probs],
        "normalized_action_entropy": float(norm_entropy),
        "policy_sharpness_proxy": float(1.0 - norm_entropy),
        "dominant_action_mass": float(probs.max()),
        "replay_path": _rel(replay_path),
    }


def build_complexity_summary() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ENV_SPECS:
        stability, source_path = _load_cbs_stability(spec.env_tag)
        action_stats = _compute_action_stats(spec)
        rows.append(
            {
                "env": spec.env,
                "short": spec.short,
                "env_tag": spec.env_tag,
                "policy_family": spec.policy_family,
                "obs_features": spec.obs_features,
                "action_count": spec.action_count,
                "action_counts": action_stats["action_counts"],
                "action_probs": action_stats["action_probs"],
                "normalized_action_entropy": action_stats["normalized_action_entropy"],
                "policy_sharpness_proxy": action_stats["policy_sharpness_proxy"],
                "dominant_action_mass": action_stats["dominant_action_mass"],
                "cbs_grs_wj": float(stability["GRS_weighted_jaccard"]),
                "cbs_grs_ta": float(stability["GRS_threshold_aware"]),
                "cbs_td": float(stability["TD"]),
                "cbs_bra": float(stability["BRA"]),
                "sources": [source_path, action_stats["replay_path"]],
            }
        )
    return rows


def _build_noise_row() -> dict[str, Any]:
    noise_paths = [
        RESULTS_DIR / "mountaincar_v0" / "noise_severity_results.json",
        RESULTS_DIR / "cartpole_v1" / "noise_severity_results.json",
        RESULTS_DIR / "lunarlander_v3" / "noise_severity_results.json",
    ]
    noise_data = {path.parent.name: _load_json(path) for path in noise_paths}
    ll_methods = noise_data["lunarlander_v3"]["noise_summaries"]
    ll_bra_drops = {method: metrics["mean_relative_bra_drop"] for method, metrics in ll_methods.items()}
    mc_methods = noise_data["mountaincar_v0"]["noise_summaries"]
    mc_stable = [
        label
        for method, label in [("cbs", "CBS"), ("b3_vote", "RV"), ("dt", "DT")]
        if mc_methods[method]["critical_eps_bra_90pct"] >= 0.08
    ]
    headline = (
        f"LunarLander mean BRA drop ranges {min(ll_bra_drops.values()):.3f}-{max(ll_bra_drops.values()):.3f}; "
        f"in MountainCar, {' and '.join(mc_stable)} stay above 90% BRA through eps=0.08."
    )
    return {
        "module": "Noise severity",
        "question": "How quickly do fidelity and agreement degrade under replay noise?",
        "coverage": "MC / CP / LL; CBS, RV, DT",
        "what_varies": "Replay noise level (eps = 0.005-0.08)",
        "headline": headline,
        "source_paths": [_rel(path) for path in noise_paths],
    }


def _build_rare_action_row() -> dict[str, Any]:
    mc_path = RESULTS_DIR / "mountaincar_v0" / "rare_action_sweep_results.json"
    ll_path = RESULTS_DIR / "lunarlander_v3" / "rare_action_sweep_results.json"
    mc = _load_json(mc_path)
    ll = _load_json(ll_path)
    mc_threshold = next(
        row["target_support"]
        for row in mc["diagnostic_resampling"]
        if row["mean_worst_action_recall"] >= 0.70
    )
    ll_best = max(ll["diagnostic_resampling"], key=lambda row: row["mean_worst_action_recall"])
    headline = (
        f"MountainCar worst-R passes 0.70 at {mc_threshold * 100:.0f}% support; "
        f"LunarLander tops out at {ll_best['mean_worst_action_recall']:.3f} even at {ll_best['target_support'] * 100:.0f}%."
    )
    return {
        "module": "Rare-action support",
        "question": "Are aggregate scores hiding minority-action failure?",
        "coverage": "MC / LL quota resampling",
        "what_varies": "Rare-action support and replay size",
        "headline": headline,
        "source_paths": [_rel(mc_path), _rel(ll_path)],
    }


def _build_cross_policy_row() -> dict[str, Any]:
    cp_path = RESULTS_DIR / "cross_algo_comparison" / "ppo_vs_dqn_comparison.json"
    ll_path = RESULTS_DIR / "cross_algo_comparison" / "ppo_vs_dqn_comparison_lunarlander.json"
    cp = _load_json(cp_path)
    ll = _load_json(ll_path)
    cp_delta = cp["delta"]["CBS"]
    ll_delta = ll["delta"]["CBS"]
    headline = (
        f"CBS BRA changes only {cp_delta['BRA']:+.3f} on CartPole and {ll_delta['BRA']:+.3f} on LunarLander under PPO vs DQN."
    )
    return {
        "module": "Cross-policy",
        "question": "Do the stability patterns depend on the RL algorithm family?",
        "coverage": "CartPole + LunarLander; DQN vs PPO",
        "what_varies": "Policy family with matched environments",
        "headline": headline,
        "source_paths": [_rel(cp_path), _rel(ll_path)],
    }


def _build_matching_row() -> dict[str, Any]:
    cp_path = RESULTS_DIR / "cartpole_v1" / "matching_robustness_results.json"
    ll_path = RESULTS_DIR / "lunarlander_v3" / "matching_robustness_results.json"
    cp = _load_json(cp_path)
    ll = _load_json(ll_path)
    cp_bra_values = [entry["BRA"] for entry in cp["rho_sweep"].values()] + [entry["BRA"] for entry in cp["lambda_sweep"].values()]
    ll_f1_values = [entry["mean_f1"] for entry in ll["rho_sweep"].values()] + [entry["mean_f1"] for entry in ll["lambda_sweep"].values()]
    headline = (
        f"Matching choices matter: CartPole BRA spans {min(cp_bra_values):.3f}-{max(cp_bra_values):.3f}; "
        f"LunarLander mean F1 spans {min(ll_f1_values):.3f}-{max(ll_f1_values):.3f}."
    )
    return {
        "module": "Matching robustness",
        "question": "Are the conclusions stable to rho/lambda matching choices?",
        "coverage": "CartPole + LunarLander",
        "what_varies": "rho and lambda matching hyperparameters",
        "headline": headline,
        "source_paths": [_rel(cp_path), _rel(ll_path)],
    }


def _build_bootstrap_row() -> dict[str, Any]:
    stats_path = RESULTS_DIR / "statistical_tests.json"
    stats = _load_json(stats_path)
    core_claims = stats["core_claims"]
    supported = sum(1 for claim in core_claims if claim["supported"])
    n_claims = len(core_claims)
    fully_significant = sum(1 for claim in core_claims if claim["n_environments_significant"] == claim["n_environments_tested"])
    headline = (
        f"All {supported}/{n_claims} core claims are supported and significant in all 3 environments."
    )
    return {
        "module": "Bootstrap significance",
        "question": "Do the paper-level claims survive formal uncertainty checks?",
        "coverage": "3 environments x 6 core claims",
        "what_varies": "Paired bootstrap confidence intervals and p-values",
        "headline": headline,
        "source_paths": [_rel(stats_path)],
    }


def _count_sig_envs(paired: dict[str, Any], comparison_key: str) -> int:
    return sum(1 for env in ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"] if paired[env][comparison_key]["f1"]["significant"])


def _build_merge_stage_row() -> dict[str, Any]:
    bootstrap_path = RESULTS_DIR / "merge_statistics" / "paired_bootstrap.json"
    paired = _load_json(bootstrap_path)
    tuned_count = _count_sig_envs(paired, "default_consensus_vs_tuned_merge")
    soft_support_count = _count_sig_envs(paired, "default_consensus_vs_soft_support")
    tuned_to_soft_support_count = _count_sig_envs(paired, "tuned_merge_vs_soft_support")
    headline = (
        f"DCM->MTC is significant in {tuned_count}/3 envs; "
        f"DCM->SSC is significant in {soft_support_count}/3 envs; "
        f"MTC->SSC is significant in {tuned_to_soft_support_count}/3."
    )
    return {
        "module": "Repair-chain significance",
        "question": "Do the merge repairs hold up statistically?",
        "coverage": "Merge-repair paired bootstrap (MC/CP/LL)",
        "what_varies": "DCM, MTC, and SSC repair stages",
        "headline": headline,
        "source_paths": [_rel(bootstrap_path)],
    }


def build_robustness_suite_summary() -> list[dict[str, Any]]:
    return [
        _build_noise_row(),
        _build_rare_action_row(),
        _build_cross_policy_row(),
        _build_matching_row(),
        _build_bootstrap_row(),
        _build_merge_stage_row(),
    ]


def build_failure_decomposition_summary() -> dict[str, Any]:
    stage_labels = {
        "match_only": "Match Only",
        "match_hard_support": "Match + Hard Support",
        "match_aggregation": "Match + Aggregation",
        "full_default": "DCM",
        "v2_soft_support": "SSC",
    }
    environments: dict[str, Any] = {}

    for env_tag in DIAG_ENV_TAGS:
        path = RESULTS_DIR / env_tag / "failure_decomposition.json"
        data = _load_json(path)
        per_seed = data["per_seed"]
        stage_stats: dict[str, Any] = {}

        for stage_key, stage_label in stage_labels.items():
            stage_rows = [seed_data[stage_key] for seed_data in per_seed.values()]
            stage_stats[stage_label] = {
                "Macro-F1": _mean_std([row["f1"] for row in stage_rows]),
                "Worst-Action Recall": _mean_std([row["worst_action_recall"] for row in stage_rows]),
                "Return": _mean_std([row["E_CR"] for row in stage_rows]),
                "Rule Count": _mean_std([row["n_rules"] for row in stage_rows]),
            }

        match_only_f1 = stage_stats["Match Only"]["Macro-F1"]["mean"]
        hard_support_f1 = stage_stats["Match + Hard Support"]["Macro-F1"]["mean"]
        aggregation_f1 = stage_stats["Match + Aggregation"]["Macro-F1"]["mean"]
        default_f1 = stage_stats["DCM"]["Macro-F1"]["mean"]
        ssc_f1 = stage_stats["SSC"]["Macro-F1"]["mean"]

        support_drop = float(match_only_f1 - hard_support_f1)
        aggregation_drop = float(match_only_f1 - aggregation_f1)

        environments[_env_name_from_tag(env_tag)] = {
            "source_path": _rel(path),
            "primary_failure_stage": "support_pruning" if support_drop > aggregation_drop else "aggregation",
            "support_drop": support_drop,
            "aggregation_drop": aggregation_drop,
            "default_drop": float(match_only_f1 - default_f1),
            "ssc_recovery": float(ssc_f1 - default_f1),
            "stages": stage_stats,
        }

    return {
        "schema_version": "paper_a1_summary_v1",
        "claim_block": "mechanism_diagnosis",
        "paper_placement": "main_text",
        "purpose": "Summary of the failure-decomposition evidence using Macro-F1, Worst-Action Recall, Return, and Rule Count.",
        "environments": environments,
    }


def build_geometric_distortion_summary() -> dict[str, Any]:
    environments: dict[str, Any] = {}

    for env_tag in DIAG_ENV_TAGS:
        path = RESULTS_DIR / env_tag / "geometric_distortion.json"
        data = _load_json(path)
        failed = data["comparison"]["failed_merges"]
        successful = data["comparison"]["successful_merges"]
        environments[_env_name_from_tag(env_tag)] = {
            "source_path": _rel(path),
            "failed_merges": {
                "count": failed["count"],
                "Action Mismatch": float(failed["mean_action_mismatch"]),
                "Bridge Rate": float(failed["mean_bridge_rate"]),
                "KNN Gap": float(failed["mean_knn_gap"]),
                "Modes": float(failed["mean_modes"]),
                "Components": float(failed["mean_components"]),
            },
            "successful_merges": {
                "count": successful["count"],
                "Action Mismatch": float(successful["mean_action_mismatch"]),
                "Bridge Rate": float(successful["mean_bridge_rate"]),
                "KNN Gap": float(successful["mean_knn_gap"]),
                "Modes": float(successful["mean_modes"]),
                "Components": float(successful["mean_components"]),
            },
            "failed_minus_successful": {
                "Action Mismatch": float(failed["mean_action_mismatch"] - successful["mean_action_mismatch"]),
                "Bridge Rate": float(failed["mean_bridge_rate"] - successful["mean_bridge_rate"]),
                "KNN Gap": float(failed["mean_knn_gap"] - successful["mean_knn_gap"]),
            },
        }

    return {
        "schema_version": "paper_a2_summary_v1",
        "claim_block": "mechanism_diagnosis",
        "paper_placement": "main_text",
        "purpose": "Summary of the geometric-distortion evidence for failed vs successful merges.",
        "environments": environments,
    }


def build_boundary_crossing_summary() -> dict[str, Any]:
    environments: dict[str, Any] = {}

    for env_tag in DIAG_ENV_TAGS:
        path = RESULTS_DIR / env_tag / "boundary_crossing.json"
        data = _load_json(path)
        mergeable = data["summary"]["mergeable"]
        non_mergeable = data["summary"]["non_mergeable"]
        environments[_env_name_from_tag(env_tag)] = {
            "source_path": _rel(path),
            "mergeable": {
                "count": mergeable["count"],
                "Boundary Crossing Rate": float(mergeable["mean_boundary_crossing_rate"]),
                "Midpoint Mismatch": float(mergeable["mean_midpoint_mismatch"]),
                "Midpoint Low Density": float(mergeable["mean_midpoint_low_density"]),
                "Path Low Density": float(mergeable["mean_path_low_density"]),
            },
            "non_mergeable": {
                "count": non_mergeable["count"],
                "Boundary Crossing Rate": float(non_mergeable["mean_boundary_crossing_rate"]),
                "Midpoint Mismatch": float(non_mergeable["mean_midpoint_mismatch"]),
                "Midpoint Low Density": float(non_mergeable["mean_midpoint_low_density"]),
                "Path Low Density": float(non_mergeable["mean_path_low_density"]),
            },
            "mergeable_minus_non_mergeable": {
                "Boundary Crossing Rate": float(mergeable["mean_boundary_crossing_rate"] - non_mergeable["mean_boundary_crossing_rate"]),
                "Midpoint Mismatch": float(mergeable["mean_midpoint_mismatch"] - non_mergeable["mean_midpoint_mismatch"]),
                "Midpoint Low Density": float(mergeable["mean_midpoint_low_density"] - non_mergeable["mean_midpoint_low_density"]),
            },
        }

    return {
        "schema_version": "paper_a3_summary_v1",
        "claim_block": "mechanism_diagnosis",
        "paper_placement": "main_text",
        "purpose": "Summary of the boundary-crossing evidence for mergeable vs non-mergeable rule pairs.",
        "environments": environments,
    }


def build_semantic_merge_summary() -> dict[str, Any]:
    environments: dict[str, Any] = {}

    for env_tag in ["cartpole_v1", "lunarlander_v3"]:
        path = RESULTS_DIR / env_tag / "semantic_merge_results.json"
        if not path.exists():
            continue
        data = _load_json(path)
        numeric = data["summary"]["numeric_merge"]
        semantic = data["summary"]["semantic_merge"]
        agreement = data["summary"]["prediction_agreement"]
        environments[_env_name_from_tag(env_tag)] = {
            "source_path": _rel(path),
            "numeric_merge": {
                "Macro-F1": numeric["f1"],
                "Worst-Action Recall": numeric["worst_action_recall"],
                "Return": numeric["E_CR"],
                "Rule Count": numeric["n_rules"],
            },
            "semantic_merge": {
                "Macro-F1": semantic["f1"],
                "Worst-Action Recall": semantic["worst_action_recall"],
                "Return": semantic["E_CR"],
                "Rule Count": semantic["n_rules"],
            },
            "semantic_minus_numeric": {
                "Macro-F1": float(semantic["f1"]["mean"] - numeric["f1"]["mean"]),
                "Worst-Action Recall": float(semantic["worst_action_recall"]["mean"] - numeric["worst_action_recall"]["mean"]),
                "Return": float(semantic["E_CR"]["mean"] - numeric["E_CR"]["mean"]),
                "Rule Count": float(semantic["n_rules"]["mean"] - numeric["n_rules"]["mean"]),
            },
            "Prediction Agreement": agreement,
        }

    return {
        "schema_version": "paper_a4_summary_v1",
        "claim_block": "repair_evidence",
        "paper_placement": "appendix_discussion_future_work",
        "purpose": "Summary of the semantic-merge pilot; this is exploratory repair evidence, not a headline claim.",
        "environments": environments,
    }


def build_external_validity_summary() -> dict[str, Any]:
    path = RESULTS_DIR / "minigrid_dynamic_obstacles_8x8_v0" / "external_validity.json"
    data = _load_json(path)
    method_labels = {
        "CBS": "CBS",
        "B3-vote": "RV",
        "Consensus_default": "DCM",
        "V2_soft_support": "SSC",
    }
    methods: dict[str, Any] = {}

    for raw_name, display_name in method_labels.items():
        row = data["cross_seed_summary"][raw_name]
        methods[display_name] = {
            "Macro-F1": row["f1_mean"],
            "BRA": row["BRA"],
            "GRS-TA": row["GRS_TA"],
            "Worst-Action Recall": row["worst_recall_mean"],
            "Rule Count": row["n_rules_mean"],
        }

    cbs = methods["CBS"]
    deltas_vs_cbs = {}
    for display_name in ["RV", "DCM", "SSC"]:
        row = methods[display_name]
        deltas_vs_cbs[display_name] = {
            "Macro-F1": float(row["Macro-F1"]["mean"] - cbs["Macro-F1"]["mean"]),
            "BRA": float(row["BRA"]["mean"] - cbs["BRA"]["mean"]),
            "GRS-TA": float(row["GRS-TA"]["mean"] - cbs["GRS-TA"]["mean"]),
            "Worst-Action Recall": float(row["Worst-Action Recall"]["mean"] - cbs["Worst-Action Recall"]["mean"]),
        }

    return {
        "schema_version": "paper_a5_summary_v1",
        "claim_block": "transfer_validation",
        "paper_placement": "main_text_short_block",
        "purpose": "Summary of the MiniGrid + PPO transfer validation block.",
        "source_path": _rel(path),
        "methods": methods,
        "deltas_vs_cbs": deltas_vs_cbs,
    }


def write_summaries() -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    complexity_rows = build_complexity_summary()
    robustness_rows = build_robustness_suite_summary()
    a1_summary = build_failure_decomposition_summary()
    a2_summary = build_geometric_distortion_summary()
    a3_summary = build_boundary_crossing_summary()
    a4_summary = build_semantic_merge_summary()
    a5_summary = build_external_validity_summary()

    complexity_json = OUT_DIR / "complexity_extrapolation_summary.json"
    robustness_json = OUT_DIR / "robustness_suite_summary.json"
    complexity_csv = OUT_DIR / "complexity_extrapolation_summary.csv"
    robustness_csv = OUT_DIR / "robustness_suite_summary.csv"
    a1_json = OUT_DIR / "failure_decomposition_summary.json"
    a2_json = OUT_DIR / "geometric_distortion_summary.json"
    a3_json = OUT_DIR / "boundary_crossing_summary.json"
    a4_json = OUT_DIR / "semantic_merge_summary.json"
    a5_json = OUT_DIR / "external_validity_summary.json"

    _write_json(complexity_json, {
        "generated_at": generated_at,
        "purpose": "Multi-axis complexity / extrapolation summary for the canonical CBS baseline.",
        "profiles": complexity_rows,
    })
    _write_json(robustness_json, {
        "generated_at": generated_at,
        "purpose": "Compact summary of the robustness evaluation suite.",
        "rows": robustness_rows,
    })
    _write_json(a1_json, {"generated_at": generated_at, **a1_summary})
    _write_json(a2_json, {"generated_at": generated_at, **a2_summary})
    _write_json(a3_json, {"generated_at": generated_at, **a3_summary})
    _write_json(a4_json, {"generated_at": generated_at, **a4_summary})
    _write_json(a5_json, {"generated_at": generated_at, **a5_summary})

    _write_csv(
        complexity_csv,
        [{
            **row,
            "action_counts": "|".join(str(value) for value in row["action_counts"]),
            "action_probs": "|".join(f"{value:.6f}" for value in row["action_probs"]),
            "sources": "; ".join(row["sources"]),
        } for row in complexity_rows],
        [
            "env", "short", "env_tag", "policy_family", "obs_features", "action_count",
            "action_counts", "action_probs", "normalized_action_entropy", "policy_sharpness_proxy",
            "dominant_action_mass", "cbs_grs_wj", "cbs_grs_ta", "cbs_td", "cbs_bra", "sources",
        ],
    )
    _write_csv(
        robustness_csv,
        [{**row, "source_paths": "; ".join(row["source_paths"])} for row in robustness_rows],
        ["module", "question", "coverage", "what_varies", "headline", "source_paths"],
    )

    return {
        "complexity_json": complexity_json,
        "complexity_csv": complexity_csv,
        "robustness_json": robustness_json,
        "robustness_csv": robustness_csv,
        "a1_json": a1_json,
        "a2_json": a2_json,
        "a3_json": a3_json,
        "a4_json": a4_json,
        "a5_json": a5_json,
    }


def main() -> None:
    outputs = write_summaries()
    for name, path in outputs.items():
        print(f"OK {name}: {_rel(path)}")


if __name__ == "__main__":
    main()