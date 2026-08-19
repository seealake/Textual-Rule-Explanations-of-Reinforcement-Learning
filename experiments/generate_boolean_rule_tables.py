#!/usr/bin/env python
"""
Generate Boolean-rule table artifacts for repository traceability.

Outputs (under experiments/results/):
1) main_comparison.{csv,json,md}
   - Methods: CBS, FT-CBS, RV, DCM, DT, BDR
    - Columns: Macro-F1, Return, GRS, GRS-TA, BRA, TD, Rule Count

2) tree_vs_boolean_baselines.{csv,json,md}
   - Methods: DT, BDR
   - Columns: Macro-F1, GRS, BRA, TD, Rule Count
"""

import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "experiments" / "results"

ENVS = [
    ("mountaincar_v0", "MountainCar-v0"),
    ("cartpole_v1", "CartPole-v1"),
    ("lunarlander_v3", "LunarLander-v3"),
]

METHOD_SPECS = [
    {
        "display": "CBS",
        "file": "stress_test_results.json",
        "key": "cbs",
    },
    {
        "display": "FT-CBS",
        "file": "stress_test_results.json",
        "key": "cbs_maxf1",
    },
    {
        "display": "RV",
        "file": "consensus_merge_results.json",
        "key": "consensus_vote",
    },
    {
        "display": "DCM",
        "file": "consensus_merge_results.json",
        "key": "consensus_cbs",
    },
    {
        "display": "DT",
        "file": "decision_tree_results.json",
        "key": "b4_dt",
    },
    {
        "display": "BDR",
        "file": "boolean_rule_results.json",
        "key": "b5_bdr",
    },
]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_metric(per_run: dict, getter) -> float:
    return float(mean(getter(run) for run in per_run.values()))


def _compute_row(env_slug: str, env_name: str, spec: dict) -> dict:
    p = RESULTS_ROOT / env_slug / spec["file"]
    data = _load_json(p)
    method = data[spec["key"]]
    per_run = method["per_run"]
    stability = method["stability"]

    row = {
        "Env": env_name,
        "Method": spec["display"],
        "Macro-F1": _mean_metric(per_run, lambda r: r["fidelity_heldout"]["f1"]),
        "Return": _mean_metric(per_run, lambda r: r["deployment"]["E_CR"]),
        "GRS": float(stability["GRS_weighted_jaccard"]),
        "GRS-TA": float(stability["GRS_threshold_aware"]),
        "BRA": float(stability["BRA"]),
        "TD": float(stability["TD"]),
        "Rule Count": _mean_metric(per_run, lambda r: r["n_rules"]),
    }
    return row


def _round_row(row: dict) -> dict:
    out = dict(row)
    for k in ["Macro-F1", "GRS", "GRS-TA", "BRA", "TD"]:
        out[k] = round(out[k], 3)
    out["Return"] = round(out["Return"], 1)
    out["Rule Count"] = round(out["Rule Count"], 1)
    return out


def _write_csv(path: Path, rows: list, columns: list):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_md(path: Path, rows: list, columns: list, title: str):
    lines = [f"# {title}", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_main_comparison() -> list:
    rows = []
    for env_slug, env_name in ENVS:
        for spec in METHOD_SPECS:
            rows.append(_round_row(_compute_row(env_slug, env_name, spec)))
    return rows


def build_non_cbs_dt_vs_bdr(main_rows: list) -> list:
    keep = {"DT", "BDR"}
    rows = []
    for row in main_rows:
        if row["Method"] in keep:
            rows.append({
                "Env": row["Env"],
                "Method": row["Method"],
                "Macro-F1": row["Macro-F1"],
                "GRS": row["GRS"],
                "BRA": row["BRA"],
                "TD": row["TD"],
                "Rule Count": row["Rule Count"],
            })
    return rows


def main():
    out_dir = RESULTS_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: main comparison including the Boolean-rule row.
    main_rows = build_main_comparison()
    main_cols = ["Env", "Method", "Macro-F1", "Return", "GRS", "GRS-TA", "BRA", "TD", "Rule Count"]

    _write_csv(out_dir / "main_comparison.csv", main_rows, main_cols)
    _write_json(
        out_dir / "main_comparison.json",
        {
            "schema_version": "main_comparison_v1",
            "columns": main_cols,
            "rows": main_rows,
        },
    )
    _write_md(out_dir / "main_comparison.md", main_rows, main_cols,
              "Main Comparison With B5")

    # Table 2: Non-CBS baselines DT vs BDR.
    non_cbs_rows = build_non_cbs_dt_vs_bdr(main_rows)
    non_cbs_cols = ["Env", "Method", "Macro-F1", "GRS", "BRA", "TD", "Rule Count"]

    _write_csv(out_dir / "tree_vs_boolean_baselines.csv", non_cbs_rows, non_cbs_cols)
    _write_json(
        out_dir / "tree_vs_boolean_baselines.json",
        {
            "schema_version": "tree_vs_boolean_baselines_v1",
            "columns": non_cbs_cols,
            "rows": non_cbs_rows,
        },
    )
    _write_md(out_dir / "tree_vs_boolean_baselines.md", non_cbs_rows, non_cbs_cols,
              "Non-CBS Baselines: DT vs BDR")

    print("Saved:")
    print("- experiments/results/main_comparison.{csv,json,md}")
    print("- experiments/results/tree_vs_boolean_baselines.{csv,json,md}")


if __name__ == "__main__":
    main()
