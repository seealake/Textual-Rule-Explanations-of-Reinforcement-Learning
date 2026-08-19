#!/usr/bin/env python
"""Recompute main-table summaries from distinct stored configurations.

The original aggregate files include method-specific configuration slots.
The main comparison uses only the 18 shared data/noise
configurations for every method.  MiniGrid accuracy is also averaged from
the stored per-run records so that all four methods use one fidelity metric.
This script performs no extraction or policy evaluation; it only
re-aggregates recorded JSON results.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUTPUT = RESULTS / "main_results.json"

ENVS = ("mountaincar_v0", "cartpole_v1", "lunarlander_v3")
SHARED_FAMILIES = {"seed_shift", "subsample", "stratified", "feature_noise"}
METHODS = {
    "CBS": ("stress_test_results.json", "cbs"),
    "FT-CBS": ("stress_test_results.json", "cbs_maxf1"),
    "DCM": ("consensus_merge_results.json", "consensus_cbs"),
    "RV": ("consensus_merge_results.json", "consensus_vote"),
    "DT": ("decision_tree_results.json", "b4_dt"),
    "BDR": ("boolean_rule_results.json", "b5_bdr"),
}

ROOT_KEYS = {
    "GRS": "GRS_weighted_jaccard",
    "GRS_TA": "GRS_threshold_aware",
    "BRA": "BRA",
    "TD": "TD",
}
PROXY_KEYS = {
    "GRS": "GRS_wj",
    "GRS_TA": "GRS_ta",
    "BRA": "BRA",
    "TD": "TD",
}


def _pairwise_mean_without_runs(method: dict, excluded: list[dict], metric: str) -> float:
    """Recover a pairwise mean after removing one complete run family.

    Each stored global proxy is the mean similarity (or distance for TD)
    from that run to all other runs.  Each family proxy is the corresponding
    mean within its family.  These row sums are sufficient to subtract every
    pair that touches the excluded family exactly.
    """
    excluded_families = {run["perturbation_family"] for run in excluded}
    if len(excluded_families) != 1:
        raise ValueError(
            "Pairwise subtraction requires one complete excluded family; "
            f"got {sorted(excluded_families)}"
        )

    n_total = int(method["stability"]["n_perturbation_runs"])
    n_excluded = len(excluded)
    n_kept = n_total - n_excluded
    root_key = ROOT_KEYS[metric]
    proxy_key = PROXY_KEYS[metric]

    total_pair_sum = float(method["stability"][root_key]) * comb(n_total, 2)
    within_excluded = (
        sum(float(run["stability_proxy_family"][proxy_key]) for run in excluded)
        * (n_excluded - 1)
        / 2.0
    )
    excluded_row_sum = sum(
        float(run["stability_proxy_global"][proxy_key]) * (n_total - 1)
        for run in excluded
    )
    kept_pair_sum = total_pair_sum - excluded_row_sum + within_excluded
    return kept_pair_sum / comb(n_kept, 2)


def _summarize_method(method: dict) -> dict:
    runs = list(method["per_run"].values())
    kept = [run for run in runs if run["perturbation_family"] in SHARED_FAMILIES]
    excluded = [run for run in runs if run["perturbation_family"] not in SHARED_FAMILIES]
    if len(kept) != 18 or not excluded:
        raise ValueError(
            f"Expected 18 kept runs and one excluded family, got "
            f"{len(kept)} kept and {len(excluded)} excluded"
        )

    summary = {
        "n_configurations": len(kept),
        "families": sorted(SHARED_FAMILIES),
        "excluded_run_ids": [run["run_id"] for run in excluded],
        "E_F1": float(np.mean([run["fidelity_heldout"]["f1"] for run in kept])),
        "worst_action_recall": float(np.mean([
            min(v["recall"] for v in run["fidelity_per_action"]["per_action"].values())
            for run in kept
        ])),
        "return": float(np.mean([run["deployment"]["E_CR"] for run in kept])),
        "rules": float(np.mean([run["n_rules"] for run in kept])),
        "by_family": {},
    }
    for family in sorted(SHARED_FAMILIES):
        family_runs = [run for run in kept if run["perturbation_family"] == family]
        summary["by_family"][family] = {
            "n_configurations": len(family_runs),
            "E_F1": float(np.mean([
                run["fidelity_heldout"]["f1"] for run in family_runs
            ])),
            "worst_action_recall": float(np.mean([
                min(v["recall"] for v in run["fidelity_per_action"]["per_action"].values())
                for run in family_runs
            ])),
        }
    for metric in ROOT_KEYS:
        summary[metric] = _pairwise_mean_without_runs(method, excluded, metric)
    return summary


def _summarize_minigrid() -> dict:
    """Average stored MiniGrid accuracy and reported stability summaries."""
    path = RESULTS / "minigrid_dynamic_obstacles_8x8_v0" / "external_validity.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    method_keys = {
        "CBS": "CBS",
        "DCM": "Consensus_default",
        "Soft support": "V2_soft_support",
        "RV": "B3-vote",
    }
    output = {}
    for display_name, stored_name in method_keys.items():
        runs = [
            run
            for outer in data["per_outer_seed"].values()
            for run in outer[stored_name]["runs"]
        ]
        stored_summary = data["cross_seed_summary"][stored_name]
        output[display_name] = {
            "n_outer_seeds": len(data["per_outer_seed"]),
            "n_runs": len(runs),
            "accuracy": float(np.mean([run["accuracy"] for run in runs])),
            "BRA": float(stored_summary["BRA"]["mean"]),
            "worst_action_recall": float(
                stored_summary["worst_recall_mean"]["mean"]
            ),
        }
    return output


def main() -> None:
    output = {
        "schema_version": "main_results_v1",
        "description": (
            "Descriptive re-aggregation of the 18 shared data/noise "
            "configurations plus MiniGrid accuracy from stored runs; no policy "
            "training, rule extraction, or policy evaluation is run."
        ),
        "environments": {},
    }

    for env in ENVS:
        output["environments"][env] = {}
        cache: dict[str, dict] = {}
        for method_name, (filename, key) in METHODS.items():
            if filename not in cache:
                with (RESULTS / env / filename).open(encoding="utf-8") as handle:
                    cache[filename] = json.load(handle)
            summary = _summarize_method(cache[filename][key])

            # The saved RV rule structure is only the first ensemble member.
            # Its ensemble-level structural quantities are therefore undefined.
            if method_name == "RV":
                summary["GRS"] = None
                summary["GRS_TA"] = None
                summary["TD"] = None
                summary["rules"] = None
            output["environments"][env][method_name] = summary

    output["minigrid"] = _summarize_minigrid()

    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
