#!/usr/bin/env python
"""
Table Generation

Generates human-readable and structured table artifacts for the paper from
stress test and consensus results.

Usage:
    python experiments/generate_summary_tables.py
    python experiments/generate_summary_tables.py --text
    python experiments/generate_summary_tables.py --json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
ENV_LABELS = {"MountainCar-v0": "MC", "CartPole-v1": "CP", "LunarLander-v3": "LL"}
RESULTS_DIR = "experiments/results"


def load_results(env_name):
    """Load stress-test, consensus, and decision-tree results for an environment."""
    tag = env_name.replace("-", "_").lower()
    results = {}

    st_path = os.path.join(RESULTS_DIR, tag, "stress_test_results.json")
    if os.path.exists(st_path):
        with open(st_path) as f:
            results["stress_test"] = json.load(f)

    cc_path = os.path.join(RESULTS_DIR, tag, "consensus_merge_results.json")
    if os.path.exists(cc_path):
        with open(cc_path) as f:
            results["consensus"] = json.load(f)

    b4_path = os.path.join(RESULTS_DIR, tag, "decision_tree_results.json")
    if os.path.exists(b4_path):
        with open(b4_path) as f:
            results["b4"] = json.load(f)

    return results


def extract_per_run_metrics(per_run_data):
    """Extract per-run F1, E_CR, worst-action recall, n_rules from per_run dict."""
    f1s, ecrs, worst_recalls, n_rules_list = [], [], [], []
    for key, run in per_run_data.items():
        f1s.append(run["fidelity_heldout"]["f1"])
        ecrs.append(run["deployment"]["E_CR"])
        n_rules_list.append(run["n_rules"])

        # Worst-action recall
        pa = run.get("fidelity_per_action", {}).get("per_action", {})
        if pa:
            recalls = [pa[a]["recall"] for a in pa]
            worst_recalls.append(min(recalls) if recalls else 0.0)
        else:
            worst_recalls.append(None)

    return {
        "f1": np.array(f1s),
        "ecr": np.array(ecrs),
        "worst_recall": np.array([r for r in worst_recalls if r is not None]),
        "n_rules": np.array(n_rules_list),
    }


def fmt(vals, fmt_str=".3f"):
    """Format mean ± std."""
    return f"{np.mean(vals):{fmt_str}} ± {np.std(vals):{fmt_str}}"


def build_table1_lines():
    """Table 1: Main stability comparison across methods."""
    lines = [
        "=" * 80,
        "TABLE 1: Main Stability Comparison",
        "=" * 80,
    ]

    header = (f"{'Env':<4} {'Method':<18} {'Macro-F1':>12} {'Worst-Action Recall':>20} "
              f"{'Return':>14} {'GRS':>8} {'GRS-TA':>10} {'BRA':>8} "
              f"{'TD':>8} {'Rule Count':>10}")
    lines.append(header)
    lines.append("-" * len(header))

    for env_name in ENVS:
        env_label = ENV_LABELS[env_name]
        data = load_results(env_name)
        if not data:
            continue

        rows = []

        # CBS
        if "stress_test" in data and "cbs" in data["stress_test"]:
            st = data["stress_test"]
            m = extract_per_run_metrics(st["cbs"]["per_run"])
            s = st["cbs"]["stability"]
            rows.append(("CBS", m, s))

        # CBS+MaxF1
        if "stress_test" in data and "cbs_maxf1" in data["stress_test"]:
            st = data["stress_test"]
            m = extract_per_run_metrics(st["cbs_maxf1"]["per_run"])
            s = st["cbs_maxf1"]["stability"]
            rows.append(("FT-CBS", m, s))

        # Consensus CBS
        if "consensus" in data and "consensus_cbs" in data["consensus"]:
            cc = data["consensus"]
            m = extract_per_run_metrics(cc["consensus_cbs"]["per_run"])
            s = cc["consensus_cbs"]["stability"]
            rows.append(("DCM", m, s))

        # rule-set voting
        if "consensus" in data and "consensus_vote" in data["consensus"]:
            cc = data["consensus"]
            m = extract_per_run_metrics(cc["consensus_vote"]["per_run"])
            s = cc["consensus_vote"]["stability"]
            rows.append(("RV", m, s))

        # Decision Tree
        if "b4" in data and "b4_dt" in data["b4"]:
            b4 = data["b4"]
            m = extract_per_run_metrics(b4["b4_dt"]["per_run"])
            s = b4["b4_dt"]["stability"]
            rows.append(("DT", m, s))

        for method_name, m, s in rows:
            wr = fmt(m["worst_recall"], ".3f") if len(m["worst_recall"]) > 0 else "N/A"
            lines.append(
                f"{env_label:<4} {method_name:<18} "
                f"{fmt(m['f1']):>12} {wr:>10} "
                f"{fmt(m['ecr'], '.1f'):>14} "
                f"{s['GRS_weighted_jaccard']:>8.4f} "
                f"{s['GRS_threshold_aware']:>8.4f} "
                f"{s['BRA']:>8.4f} "
                f"{s['TD']:>8.4f} "
                f"{fmt(m['n_rules'], '.1f'):>10}"
            )
        lines.append("")

    return lines


def build_table2_lines():
    """Table 2: Perturbation source decomposition."""
    lines = [
        "=" * 80,
        "TABLE 2: Perturbation Source Decomposition",
        "=" * 80,
    ]

    families = {
        "seed_shift": "seed_shift_s",
        "subsample": "subsample_",
        "stratified": "stratified_",
        "cluster_count": "cluster_delta_",
        "feature_noise": "noise_",
    }

    for env_name in ENVS:
        env_label = ENV_LABELS[env_name]
        data = load_results(env_name)
        if not data:
            continue

        lines.append("")
        lines.append(f"--- {env_name} ---")
        header = f"{'Family':<16} {'Method':<18} {'Macro-F1':>8} {'Return':>10} {'Rule Count':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        methods_data = []
        if "stress_test" in data:
            methods_data.append(("CBS", data["stress_test"].get("cbs", {}).get("per_run", {})))
            methods_data.append(("FT-CBS", data["stress_test"].get("cbs_maxf1", {}).get("per_run", {})))
        if "consensus" in data:
            methods_data.append(("DCM", data["consensus"].get("consensus_cbs", {}).get("per_run", {})))
        if "b4" in data:
            methods_data.append(("DT", data["b4"].get("b4_dt", {}).get("per_run", {})))

        for family_name, prefix in families.items():
            for method_name, per_run in methods_data:
                fam_runs = {k: v for k, v in per_run.items() if k.startswith(prefix)}
                if not fam_runs:
                    continue
                m = extract_per_run_metrics(fam_runs)
                lines.append(
                    f"{family_name:<16} {method_name:<18} "
                    f"{np.mean(m['f1']):>8.4f} "
                    f"{np.mean(m['ecr']):>10.1f} "
                    f"{np.mean(m['n_rules']):>6.1f}"
                )
            lines.append("")

    return lines


def build_table3_lines():
    """Table 3: B × τ ablation."""
    lines = [
        "=" * 80,
        "TABLE 3: B × τ Ablation",
        "=" * 80,
    ]

    for env_name in ENVS:
        env_label = ENV_LABELS[env_name]
        data = load_results(env_name)
        if not data or "consensus" not in data:
            continue

        abl = data["consensus"].get("ablations", {}).get("B_tau_grid", {})
        if not abl:
            continue

        lines.append("")
        lines.append(f"--- {env_name} ---")
        lines.append(f"{'':>15} {'τ=0.5':>20} {'τ=0.7':>20} {'τ=0.9':>20}")
        lines.append(f"{'':>15} {'F1 / GRS / rules':>20} {'F1 / GRS / rules':>20} {'F1 / GRS / rules':>20}")
        lines.append("-" * 75)

        for B in [3, 5, 10]:
            row = f"  B={B:<10}"
            for tau in [0.5, 0.7, 0.9]:
                key = f"B{B}_tau{tau}"
                if key in abl:
                    cell = abl[key]
                    f1 = cell["fidelity"]["mean_f1"]
                    grs = cell["stability"]["GRS_wj"]
                    rules = cell["mean_consensus_rules"]
                    row += f"  {f1:.3f}/{grs:.3f}/{rules:.0f}  "
                else:
                    row += f"  {'N/A':>18}  "
            lines.append(row)

        # Also print worst-action recall
        lines.append("")
        lines.append("  Worst-action recall:")
        for B in [3, 5, 10]:
            row = f"  B={B:<10}"
            for tau in [0.5, 0.7, 0.9]:
                key = f"B{B}_tau{tau}"
                if key in abl:
                    cell = abl[key]
                    wr = cell["fidelity"]["mean_worst_action_recall"]
                    row += f"  {wr:>18.4f}  "
                else:
                    row += f"  {'N/A':>18}  "
            lines.append(row)

        # Print matching hyperparameter sweeps
        for sweep_name, sweep_data in [
            ("ρ sweep", data["consensus"].get("ablations", {}).get("rho_sweep", {})),
            ("λ sweep", data["consensus"].get("ablations", {}).get("lambda_sweep", {})),
        ]:
            if sweep_data:
                lines.append("")
                lines.append(f"  {sweep_name}:")
                for label, cell in sweep_data.items():
                    f1 = cell["fidelity"]["mean_f1"]
                    grs = cell["stability"]["GRS_wj"]
                    bra = cell["stability"]["BRA"]
                    lines.append(f"    {label}: F1={f1:.4f}, GRS={grs:.4f}, BRA={bra:.4f}")

    return lines


def build_tables_text():
    """Build the full human-readable table report."""
    lines = []
    for builder in (build_table1_lines, build_table2_lines, build_table3_lines):
        if lines:
            lines.append("")
        lines.extend(builder())
    lines.append("")
    lines.append("=" * 80)
    lines.append("All tables generated.")
    return "\n".join(lines) + "\n"


def write_text_tables(out_path):
    """Write the human-readable table report to disk."""
    report = build_tables_text()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


def generate_summary_tables_json():
    """Generate all table data as a structured JSON dict."""
    output = {"schema_version": "tables_v1", "tables": {}}

    for env_name in ENVS:
        env_label = ENV_LABELS[env_name]
        data = load_results(env_name)
        if not data:
            continue

        env_key = env_name.replace("-", "_").lower()

        # Table 1: Main comparison
        table1_rows = []
        method_sources = [
            ("CBS", "stress_test", "cbs"),
            ("FT-CBS", "stress_test", "cbs_maxf1"),
            ("DCM", "consensus", "consensus_cbs"),
            ("RV", "consensus", "consensus_vote"),
            ("DT", "b4", "b4_dt"),
        ]
        for method_name, source_key, method_key in method_sources:
            if source_key not in data:
                continue
            block = data[source_key].get(method_key)
            if not block:
                continue
            m = extract_per_run_metrics(block["per_run"])
            s = block["stability"]
            row = {
                "method": method_name,
                "F1_mean": float(np.mean(m["f1"])),
                "F1_std": float(np.std(m["f1"])),
                "worst_recall_mean": float(np.mean(m["worst_recall"])) if len(m["worst_recall"]) > 0 else None,
                "worst_recall_std": float(np.std(m["worst_recall"])) if len(m["worst_recall"]) > 0 else None,
                "E_CR_mean": float(np.mean(m["ecr"])),
                "E_CR_std": float(np.std(m["ecr"])),
                "GRS_wj": s["GRS_weighted_jaccard"],
                "GRS_TA": s["GRS_threshold_aware"],
                "BRA": s["BRA"],
                "TD": s["TD"],
                "n_rules_mean": float(np.mean(m["n_rules"])),
                "n_rules_std": float(np.std(m["n_rules"])),
                "n_runs": len(m["f1"]),
            }
            table1_rows.append(row)

        output["tables"].setdefault("table1_main_comparison", {})[env_key] = table1_rows

        # Table 2: Per-family breakdown
        families = {
            "seed_shift": "seed_shift_s",
            "subsample": "subsample_",
            "stratified": "stratified_",
            "cluster_count": "cluster_delta_",
            "feature_noise": "noise_",
        }
        methods_data = []
        if "stress_test" in data:
            methods_data.append(("CBS", data["stress_test"].get("cbs", {}).get("per_run", {})))
            methods_data.append(("FT-CBS", data["stress_test"].get("cbs_maxf1", {}).get("per_run", {})))
        if "consensus" in data:
            methods_data.append(("DCM", data["consensus"].get("consensus_cbs", {}).get("per_run", {})))
        if "b4" in data:
            methods_data.append(("DT", data["b4"].get("b4_dt", {}).get("per_run", {})))

        table2_rows = []
        for family_name, prefix in families.items():
            for method_name, per_run in methods_data:
                fam_runs = {k: v for k, v in per_run.items() if k.startswith(prefix)}
                if not fam_runs:
                    continue
                m = extract_per_run_metrics(fam_runs)
                table2_rows.append({
                    "family": family_name,
                    "method": method_name,
                    "F1_mean": float(np.mean(m["f1"])),
                    "E_CR_mean": float(np.mean(m["ecr"])),
                    "n_rules_mean": float(np.mean(m["n_rules"])),
                    "n_runs": len(m["f1"]),
                })
        output["tables"].setdefault("table2_perturbation_decomposition", {})[env_key] = table2_rows

        # Table 3: Ablation
        if "consensus" in data:
            abl_data = data["consensus"].get("ablations", {})
            table3 = {}

            bt_grid = abl_data.get("B_tau_grid", {})
            if bt_grid:
                table3["B_tau_grid"] = {}
                for key, cell in bt_grid.items():
                    table3["B_tau_grid"][key] = {
                        "F1": cell["fidelity"]["mean_f1"],
                        "GRS_wj": cell["stability"]["GRS_wj"],
                        "BRA": cell["stability"].get("BRA"),
                        "mean_rules": cell["mean_consensus_rules"],
                        "worst_recall": cell["fidelity"]["mean_worst_action_recall"],
                    }

            for sweep_name in ["rho_sweep", "lambda_sweep"]:
                sweep_data = abl_data.get(sweep_name, {})
                if sweep_data:
                    table3[sweep_name] = {}
                    for label, cell in sweep_data.items():
                        table3[sweep_name][label] = {
                            "F1": cell["fidelity"]["mean_f1"],
                            "GRS_wj": cell["stability"]["GRS_wj"],
                            "BRA": cell["stability"]["BRA"],
                        }

            output["tables"].setdefault("table3_ablation", {})[env_key] = table3

    return output


def main():
    parser = argparse.ArgumentParser(description="Generate tables")
    parser.add_argument("--json", action="store_true",
                        help="Write structured JSON to experiments/results/summary_tables.json")
    parser.add_argument("--text", action="store_true",
                        help="Write human-readable text to experiments/results/summary_tables.txt")
    parser.add_argument("--stdout", action="store_true",
                        help="Also print the human-readable text report to stdout")
    args = parser.parse_args()

    write_json = args.json or not (args.json or args.text)
    write_text = args.text or not (args.json or args.text)

    if write_json:
        result = generate_summary_tables_json()
        out_path = os.path.join(RESULTS_DIR, "summary_tables.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Structured tables saved to {out_path}")

    if write_text:
        out_path = os.path.join(RESULTS_DIR, "summary_tables.txt")
        report = write_text_tables(out_path)
        print(f"Human-readable tables saved to {out_path}")
        if args.stdout:
            print(report, end="")


if __name__ == "__main__":
    main()
