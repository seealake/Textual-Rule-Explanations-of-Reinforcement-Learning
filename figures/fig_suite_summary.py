#!/usr/bin/env python
"""Compact summary of the robustness suite."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from experiments.generate_suite_summary import build_robustness_suite_summary
from figures._style import apply_style, savefig, COL2


ROW_COLORS = [
    "#EAF2FF",
    "#EEF8F1",
    "#FFF3E8",
    "#F4EEFF",
    "#FFF8DE",
    "#FBEDEE",
]


def _wrap(text, width):
    return textwrap.fill(text, width=width)


def main():
    apply_style()
    rows = build_robustness_suite_summary()

    wrapped_rows = []
    for row in rows:
        wrapped = {
            "module": _wrap(row["module"], 14),
            "what_varies": _wrap(row["what_varies"], 20),
            "coverage": _wrap(row["coverage"], 18),
            "headline": _wrap(row["headline"], 42),
        }
        line_count = max(len(text.splitlines()) for text in wrapped.values())
        row_height = max(0.92, 0.24 * line_count + 0.18)
        wrapped_rows.append((wrapped, row_height))

    header_height = 0.82
    row_gap = 0.12
    footer_height = 0.34
    top_pad = 0.30
    bottom_pad = 0.22
    total_height = (
        top_pad
        + header_height
        + sum(row_height for _, row_height in wrapped_rows)
        + row_gap * (len(wrapped_rows) - 1)
        + footer_height
        + bottom_pad
    )

    fig_height = max(5.2, total_height * 0.62)
    fig, ax = plt.subplots(figsize=(COL2, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    columns = [
        ("Module", 0.03),
        ("What Varies", 0.21),
        ("Coverage", 0.44),
        ("Headline Finding", 0.63),
    ]

    current_y = total_height - top_pad
    header_y0 = current_y - header_height
    header_y = header_y0 + header_height / 2.0
    ax.add_patch(Rectangle((0.01, header_y0), 0.98, header_height, facecolor="#243447", edgecolor="none"))
    for title, xpos in columns:
        ax.text(xpos, header_y, title, color="white", fontweight="bold", va="center", ha="left")

    current_y = header_y0 - row_gap
    for row_idx, ((wrapped, row_height), row) in enumerate(zip(wrapped_rows, rows)):
        y0 = current_y - row_height
        y_mid = y0 + row_height / 2.0
        ax.add_patch(
            Rectangle(
                (0.01, y0),
                0.98,
                row_height,
                facecolor=ROW_COLORS[row_idx % len(ROW_COLORS)],
                edgecolor="#D9D9D9",
                linewidth=0.6,
            )
        )
        ax.text(columns[0][1], y_mid, wrapped["module"], va="center", ha="left", fontweight="bold", fontsize=9)
        ax.text(columns[1][1], y_mid, wrapped["what_varies"], va="center", ha="left", fontsize=9)
        ax.text(columns[2][1], y_mid, wrapped["coverage"], va="center", ha="left", fontsize=9)
        ax.text(columns[3][1], y_mid, wrapped["headline"], va="center", ha="left", fontsize=9)
        current_y = y0 - row_gap

    ax.text(
        0.01,
        bottom_pad + 0.03,
        "All rows reuse existing experiment artifacts; the goal is to show the evaluation suite as a system rather than as isolated appendices.",
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
    )
    fig.suptitle("Robustness suite summary", fontsize=9.4, y=0.985)

    savefig(fig, "fig_suite_summary")


if __name__ == "__main__":
    main()