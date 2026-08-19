"""
Shared figure style and constants for all paper figures.

Usage in each figure script:
    from _style import apply_style, COLORS, MARKERS, LABELS, load_results, ENVS, ...
    apply_style()
"""
import os
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "figures"

# ── Environments ─────────────────────────────────────────────────────
ENVS = ["MountainCar-v0", "CartPole-v1", "LunarLander-v3"]
ENV_TAGS = {"MountainCar-v0": "mountaincar_v0",
            "CartPole-v1": "cartpole_v1",
            "LunarLander-v3": "lunarlander_v3"}
ENV_SHORT = {"MountainCar-v0": "MountainCar",
             "CartPole-v1": "CartPole",
             "LunarLander-v3": "LunarLander"}

# ── Colorblind-friendly palette (IBM Design) ────────────────────────
# Canonical keys are display names; aliases keep result-file keys working
# so that scripts passing internal keys never get a KeyError.
_BASE_COLORS = {
    "CBS":     "#648FFF",   # blue
    "FT-CBS":  "#FE6100",   # orange
    "RV":      "#35A86B",   # green (adjusted for contrast)
    "DCM":     "#9B59B6",   # purple
    "DT":      "#DC3220",   # red
    "BDR":     "#785EF0",   # violet
    "MTC":     "#FFB000",   # amber
    "SSC":     "#22A7F0",   # sky blue
    "W-RV":    "#2E8B57",   # sea green
}
_BASE_MARKERS = {
    "CBS": "o", "FT-CBS": "s", "RV": "^",
    "DCM": "p", "DT": "D",
    "BDR": "v", "MTC": "P", "SSC": "X", "W-RV": "*",
}

# Keys used inside result files → display name
_ALIASES = {
    "CBS+MaxF1": "FT-CBS", "MaxF1": "FT-CBS", "cbs_maxf1": "FT-CBS",
    "B3-vote": "RV", "b3_vote": "RV", "consensus_vote": "RV",
    "Consensus CBS": "DCM", "consensus_cbs": "DCM", "default_consensus": "DCM",
    "DT Surrogate": "DT", "b4_dt": "DT", "dt": "DT", "decision_tree_surrogate": "DT",
    "B5-BDR": "BDR", "weighted_b3vote": "W-RV", "weighted_vote": "W-RV",
    "tuned_merge": "MTC", "match_threshold_check": "MTC", "v2_soft_support": "SSC",
}

# COLORS / MARKERS accept both display names and result-file keys
COLORS = dict(_BASE_COLORS)
COLORS.update({alias: _BASE_COLORS[canon]
               for alias, canon in _ALIASES.items()
               if canon in _BASE_COLORS})

MARKERS = dict(_BASE_MARKERS)
MARKERS.update({alias: _BASE_MARKERS[canon]
                for alias, canon in _ALIASES.items()
                if canon in _BASE_MARKERS})
LABELS = list(COLORS.keys())

# Method keys -> display labels (internal)
KEY_TO_LABEL = {
    "cbs": "CBS", "cbs_maxf1": "FT-CBS",
    "consensus_vote": "RV", "b3_vote": "RV",
    "consensus_cbs": "DCM",
    "b4_dt": "DT", "dt": "DT",
}

# ── Display name mappings ──────────────────────────────
# Used by figure legends, table headers, and captions.
# Internal keys (JSON, script) are preserved; only display layer changes.
METHOD_DISPLAY_NAMES = {
    "CBS": "CBS",
    "CBS+MaxF1": "FT-CBS",
    "MaxF1": "FT-CBS",
    "cbs_maxf1": "FT-CBS",
    "consensus_cbs": "DCM",
    "Consensus CBS": "DCM",
    "default_consensus": "DCM",
    "B3-vote": "RV",
    "b3_vote": "RV",
    "consensus_vote": "RV",
    "tuned_merge": "MTC",
    "match_threshold_check": "MTC",
    "v2_soft_support": "SSC",
    "DT": "DT",
    "DT Surrogate": "DT",
    "decision_tree_surrogate": "DT",
    "b4_dt": "DT",
    "dt": "DT",
    "B5-BDR": "BDR",
    "BDR": "BDR",
    "weighted_b3vote": "W-RV",
    "weighted_vote": "W-RV",
}

METRIC_DISPLAY_NAMES = {
    "F1": "Macro-F1",
    "worst_action_recall": "Worst-Action Recall",
    "worst-R": "Worst-Action Recall",
    "GRS_wj": "GRS",
    "GRS_weighted_jaccard": "GRS",
    "GRS_ta": "GRS-TA",
    "GRS_threshold_aware": "GRS-TA",
    "BRA": "BRA",
    "TD": "TD",
    "E_CR": "Return",
    "rules": "Rule Count",
}

# Canonical method display order for paper tables
METHOD_ORDER = ["CBS", "FT-CBS", "DCM", "RV", "MTC", "SSC", "DT", "BDR", "W-RV"]


def display_method(name):
    """Map an internal method name to its display name."""
    return METHOD_DISPLAY_NAMES.get(name, name)


def display_metric(name):
    """Map an internal metric name to its display name."""
    return METRIC_DISPLAY_NAMES.get(name, name)


def method_color(name, default="#999999"):
    """Return the color for any method name, display or internal."""
    return COLORS.get(name, COLORS.get(display_method(name), default))


def method_marker(name, default="o"):
    """Return the marker for any method name, display or internal."""
    return MARKERS.get(name, MARKERS.get(display_method(name), default))

# ── Typography & Layout ──────────────────────────────────────────────
# Target: single-column = 3.5in, double-column = 7.16in (IEEE/NeurIPS)
COL1 = 3.5    # single column width (inches)
COL2 = 7.16   # double column width (inches)

# ── PGF / LaTeX detection ───────────────────────────────────────────
_HAS_LATEX = (
    shutil.which("pdflatex") is not None
    and os.environ.get("MPLBACKEND_NO_LATEX") != "1"
)


def apply_style():
    """Apply unified academic figure style. Call once at script start.

    If pdflatex is found, uses the PGF backend so all text is rendered
    by LaTeX (perfect font matching with the paper body).  Otherwise
    falls back to matplotlib's built-in mathtext with Computer Modern.
    """
    if _HAS_LATEX:
        mpl.use("pgf")

    sns.set_context("paper", font_scale=1.0)
    sns.set_style("ticks", {
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    rc = {
        # Sizes (pt); illustration text stays at 9pt or larger
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 4,
        "axes.titlepad": 6,
        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.3,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 1.0,
    }

    if _HAS_LATEX:
        rc.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,          # don't override LaTeX fonts
            "text.usetex": True,
            "pgf.preamble": "\n".join([
                r"\usepackage[utf8]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{lmodern}",       # Latin Modern (improved CM)
            ]),
            "font.family": "serif",
        })
    else:
        rc.update({
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern",
                           "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
        })

    mpl.rcParams.update(rc)


def savefig(fig, name, tight=True):
    """Save figure as PDF (vector) and PNG (raster preview).

    When the PGF backend is active, matplotlib cannot render PNG
    directly.  In that case we save the PDF first and then convert
    it to PNG using PyMuPDF (``fitz``).
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    kwargs = {}
    if tight:
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = 0.02
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, **kwargs)

    if _HAS_LATEX:
        # PGF backend — convert PDF → PNG via PyMuPDF
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=300)
            pix.save(str(png_path))
            doc.close()
        except Exception as exc:
            print(f"  [WARN] PNG conversion failed ({exc}); PDF saved OK")
    else:
        fig.savefig(png_path, dpi=300, **kwargs)

    plt.close(fig)
    print(f"  OK {name}.pdf/png")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_results(env_tag, filename):
    """Load a JSON result file for a given environment tag."""
    path = RESULTS_DIR / env_tag / filename
    if path.exists():
        return load_json(path)
    return None
