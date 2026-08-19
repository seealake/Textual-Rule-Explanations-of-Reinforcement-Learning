#!/usr/bin/env python
"""Extract a compact rule-set voting summary."""
import json
import numpy as np

for env in ["mountaincar_v0", "cartpole_v1"]:
    d = json.load(open(f"experiments/results/{env}/consensus_merge_results.json"))
    vote = d["consensus_vote"]
    stab = vote["stability"]
    per_run = vote["per_run"]

    f1s, ecrs, war_list, n_rules_list = [], [], [], []
    for k, r in per_run.items():
        pa = r.get("fidelity_per_action", {})
        macro_f1 = pa.get("macro_f1", r["fidelity_heldout"]["f1"])
        f1s.append(macro_f1)
        ecrs.append(r["deployment"]["E_CR"])
        n_rules_list.append(r.get("n_rules", 0))
        pa_dict = pa.get("per_action", {})
        if pa_dict:
            recalls = [pa_dict[a]["recall"] for a in pa_dict]
            war_list.append(min(recalls))

    f1s = np.array(f1s)
    ecrs = np.array(ecrs)
    war = np.array(war_list)
    nrules = np.array(n_rules_list)

    print(f"\n=== {env} rule-set voting (fixed code, {len(per_run)} runs) ===")
    print(f"  F1:    {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"  E_CR:  {ecrs.mean():.1f} +/- {ecrs.std():.1f}")
    print(f"  worst-action recall: {war.mean():.3f} +/- {war.std():.3f}")
    print(f"  rules: {nrules.mean():.1f} +/- {nrules.std():.1f}")
    print(f"  GRS_wj: {stab['GRS_weighted_jaccard']:.4f}")
    print(f"  GRS-TA: {stab['GRS_threshold_aware']:.4f}")
    print(f"  BRA:    {stab['BRA']:.4f}")
    print(f"  TD:     {stab['TD']:.4f}")

    # CBS baseline from stress_test
    st = json.load(open(f"experiments/results/{env}/stress_test_results.json"))
    cbs_st = st["cbs"]["stability"]
    cbs_runs = st["cbs"]["per_run"]
    cbs_f1s = [r["fidelity_heldout"]["f1"] for r in cbs_runs.values()]
    cbs_ecrs = [r["deployment"]["E_CR"] for r in cbs_runs.values()]
    cbs_pa_all = []
    for r in cbs_runs.values():
        pa = r.get("fidelity_per_action", {}).get("per_action", {})
        if pa:
            cbs_pa_all.append(min(pa[a]["recall"] for a in pa))
    print(f"\n  CBS baseline:")
    print(f"    F1:    {np.mean(cbs_f1s):.3f} +/- {np.std(cbs_f1s):.3f}")
    print(f"    E_CR:  {np.mean(cbs_ecrs):.1f} +/- {np.std(cbs_ecrs):.1f}")
    if cbs_pa_all:
        print(f"    worst-action recall: {np.mean(cbs_pa_all):.3f} +/- {np.std(cbs_pa_all):.3f}")
    print(f"    GRS_wj: {cbs_st['GRS_weighted_jaccard']:.4f}")
    print(f"    GRS-TA: {cbs_st['GRS_threshold_aware']:.4f}")
    print(f"    BRA:    {cbs_st['BRA']:.4f}")
    print(f"    TD:     {cbs_st['TD']:.4f}")
