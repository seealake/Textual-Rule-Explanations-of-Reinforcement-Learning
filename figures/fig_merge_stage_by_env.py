#!/usr/bin/env python
"""Environment-specific failure modes.

Intended caption/message:
Failure mode differs by environment because policy geometry, action-space
structure, minority-action difficulty, low-density separation, and the merge
heuristic interact differently; raw state dimension alone is not enough.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from figures._style import COL2, apply_style, load_results, savefig


HEADER_BG = "#F4F1EA"
ROW_BG = "#FBFAF7"
ALT_ROW_BG = "#F7F5EF"
GRID = "#D9D4C8"
TEXT = "#1F1F1F"
SUBTLE = "#5E5A54"

STAGE_COLORS = {
    "Aggregation": "#FDE6D6",
    "Aggregation (mild)": "#F9EEDF",
    "Support pruning": "#E3ECFB",
}

METRIC_COLORS = {
    "mismatch": "#DC3220",
    "crossing": "#648FFF",
    "density": "#35A86B",
}


def _load_required(env_tag, filename):
    data = load_results(env_tag, filename)
    if data is None:
        raise FileNotFoundError(f"Missing {filename} for {env_tag}")
    return data


def _build_rows():
    mc_fd = _load_required("mountaincar_v0", "failure_decomposition.json")["summary"]
    mc_gd = _load_required("mountaincar_v0", "geometric_distortion.json")["comparison"]
    mc_bc = _load_required("mountaincar_v0", "boundary_crossing.json")["summary"]

    cp_fd = _load_required("cartpole_v1", "failure_decomposition.json")["summary"]
    cp_gd = _load_required("cartpole_v1", "geometric_distortion.json")["comparison"]
    cp_bc = _load_required("cartpole_v1", "boundary_crossing.json")["summary"]

    ll_fd = _load_required("lunarlander_v3", "failure_decomposition.json")["summary"]
    ll_gd = _load_required("lunarlander_v3", "geometric_distortion.json")["comparison"]
    ll_bc = _load_required("lunarlander_v3", "boundary_crossing.json")["summary"]

    # Validate the external-validity contrast artifact used in the footer note.
    _load_required("minigrid_dynamic_obstacles_8x8_v0", "external_validity.json")

    cp_pruned = int(round(cp_fd["match_hard_support"]["filtered_groups"]["mean"]))
    cp_matched = int(round(cp_fd["match_only"]["surviving_groups"]["mean"]))

    return [
        {
            "env": "MountainCar",
            "dim": "2",
            "actions": "3",
            "stage": "Aggregation (mild)",
            "stage_note": "smaller drop than\nCartPole/LunarLander",
            "mismatch_value": mc_gd["failed_merges"]["mean_action_mismatch"],
            "mismatch_text": f"{mc_gd['failed_merges']['mean_action_mismatch']:.2f}",
            "crossing_value": mc_bc["mergeable"]["mean_boundary_crossing_rate"],
            "crossing_text": f"{mc_bc['mergeable']['mean_boundary_crossing_rate']:.0%}",
            "density_value": None,
            "density_text": "-",
            "interpretation": "Same interval-collapse story,\nbut milder than LunarLander.",
        },
        {
            "env": "CartPole",
            "dim": "4",
            "actions": "2",
            "stage": "Support pruning",
            "stage_note": f"hard tau prunes\n{cp_pruned}/{cp_matched} groups",
            "mismatch_value": cp_gd["failed_merges"]["mean_action_mismatch"],
            "mismatch_text": f"{cp_gd['failed_merges']['mean_action_mismatch']:.2f}",
            "crossing_value": cp_bc["mergeable"]["mean_boundary_crossing_rate"],
            "crossing_text": f"{cp_bc['mergeable']['mean_boundary_crossing_rate']:.0%}",
            "density_value": cp_bc["mergeable"]["mean_midpoint_low_density"],
            "density_text": f"{cp_bc['mergeable']['mean_midpoint_low_density']:.0%} midpoint",
            "interpretation": "Cleaner 2-action structure;\nmany matched groups remain usable,\nbut hard retention removes them early.",
        },
        {
            "env": "LunarLander",
            "dim": "8",
            "actions": "4",
            "stage": "Aggregation",
            "stage_note": "dominant loss appears\nbefore support",
            "mismatch_value": ll_gd["failed_merges"]["mean_action_mismatch"],
            "mismatch_text": f"{ll_gd['failed_merges']['mean_action_mismatch']:.2f}",
            "crossing_value": ll_bc["mergeable"]["mean_boundary_crossing_rate"],
            "crossing_text": f"{ll_bc['mergeable']['mean_boundary_crossing_rate']:.0%}",
            "density_value": ll_gd["failed_merges"]["mean_bridge_rate"],
            "density_text": f"{ll_gd['failed_merges']['mean_bridge_rate']:.2f} bridge",
            "interpretation": "8D / 4 actions with minority-action difficulty;\nmerged intervals compress multi-modal,\naction-inconsistent regions.",
        },
    ]


def _draw_rect(ax, x, y, w, h, facecolor, edgecolor=GRID, linewidth=0.6):
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def _draw_text(ax, x, y, w, h, text, fontsize=9, weight="normal", color=TEXT, align="left"):
    if align == "left":
        tx = x + w * 0.05
        ha = "left"
    else:
        tx = x + w * 0.5
        ha = "center"
    ax.text(tx, y + h * 0.5, text, ha=ha, va="center", fontsize=fontsize, fontweight=weight, color=color)


def _draw_stage_cell(ax, x, y, w, h, stage, note):
    _draw_rect(ax, x, y, w, h, STAGE_COLORS.get(stage, ROW_BG))
    ax.text(x + w * 0.05, y + h * 0.34, stage, ha="left", va="center", fontsize=9, fontweight="bold", color=TEXT)
    ax.text(x + w * 0.05, y + h * 0.72, note, ha="left", va="center", fontsize=9, color=SUBTLE)


def _draw_metric_cell(ax, x, y, w, h, value, label, color):
    _draw_rect(ax, x, y, w, h, "#FFFFFF")
    if value is None:
        ax.text(x + w * 0.5, y + h * 0.5, "-", ha="center", va="center", fontsize=9, color=SUBTLE)
        return

    bar_x = x + w * 0.08
    bar_y = y + h * 0.62
    bar_w = w * 0.84
    bar_h = h * 0.12
    _draw_rect(ax, bar_x, bar_y, bar_w, bar_h, "#ECE6DA", edgecolor="none", linewidth=0.0)
    _draw_rect(ax, bar_x, bar_y, bar_w * max(0.0, min(1.0, value)), bar_h, color, edgecolor="none", linewidth=0.0)
    ax.text(x + w * 0.08, y + h * 0.34, label, ha="left", va="center", fontsize=9, fontweight="bold", color=TEXT)


def main():
    apply_style()
    rows = _build_rows()

    columns = [
        ("Environment", 1.40),
        ("State\ndim", 0.70),
        ("\\#\nActions", 0.80),
        ("Dominant failure\nstage", 2.25),
        ("Failed action\nmismatch", 1.50),
        ("Mergeable\ncrossing", 1.35),
        ("Low-density\nsignal", 1.55),
        ("Interpretation", 3.45),
    ]

    total_width = sum(width for _, width in columns)
    header_h = 0.82
    row_h = 1.05
    footer_h = 0.68
    total_height = header_h + len(rows) * row_h + footer_h

    fig, ax = plt.subplots(figsize=(COL2, 3.0))
    ax.set_xlim(0, total_width)
    ax.set_ylim(total_height, 0)
    ax.axis("off")

    x_positions = [0.0]
    for _, width in columns[:-1]:
        x_positions.append(x_positions[-1] + width)

    y = 0.0
    x = 0.0
    for header, width in columns:
        _draw_rect(ax, x, y, width, header_h, HEADER_BG)
        _draw_text(ax, x, y, width, header_h, header, fontsize=9, weight="bold", align="center")
        x += width

    for row_idx, row in enumerate(rows):
        y = header_h + row_idx * row_h
        base_bg = ROW_BG if row_idx % 2 == 0 else ALT_ROW_BG

        x = 0.0
        _draw_rect(ax, x, y, columns[0][1], row_h, base_bg)
        _draw_text(ax, x, y, columns[0][1], row_h, row["env"], fontsize=9, weight="bold")
        x += columns[0][1]

        _draw_rect(ax, x, y, columns[1][1], row_h, base_bg)
        _draw_text(ax, x, y, columns[1][1], row_h, row["dim"], fontsize=9, weight="bold", align="center")
        x += columns[1][1]

        _draw_rect(ax, x, y, columns[2][1], row_h, base_bg)
        _draw_text(ax, x, y, columns[2][1], row_h, row["actions"], fontsize=9, weight="bold", align="center")
        x += columns[2][1]

        _draw_stage_cell(ax, x, y, columns[3][1], row_h, row["stage"], row["stage_note"])
        x += columns[3][1]

        _draw_metric_cell(ax, x, y, columns[4][1], row_h, row["mismatch_value"], row["mismatch_text"], METRIC_COLORS["mismatch"])
        x += columns[4][1]

        _draw_metric_cell(ax, x, y, columns[5][1], row_h, row["crossing_value"], row["crossing_text"], METRIC_COLORS["crossing"])
        x += columns[5][1]

        _draw_metric_cell(ax, x, y, columns[6][1], row_h, row["density_value"], row["density_text"], METRIC_COLORS["density"])
        x += columns[6][1]

        _draw_rect(ax, x, y, columns[7][1], row_h, base_bg)
        _draw_text(ax, x, y, columns[7][1], row_h, row["interpretation"], fontsize=9)

    footer_text = (
        "Transfer contrast: MiniGrid (14D, 3 actions, PPO) remains more stable despite higher dimension;\n"
        "the story is geometry + action semantics + merge behavior, not dimension alone."
    )
    ax.text(0.02, total_height - 0.16, footer_text, ha="left", va="top", fontsize=9, color=SUBTLE, style="italic")

    fig.suptitle("Environment-Specific Failure Modes in Default Merge", fontsize=9, y=1.02)
    plt.tight_layout()
    savefig(fig, "fig_merge_stage_by_env")


if __name__ == "__main__":
    main()