"""Shared look for every figure in stats/ -- palette, sample markers, savefig.

Why a module instead of per-script styling: the bar charts, the atlas maps and
the effect-size plots get shown side by side on one slide. If Control is blue in
one and orange in the next, the audience reads a difference that is not there.

Colour rules used here, and the reason for each:

* Groups are CATEGORICAL -> two fixed hues, assigned by group letter (a, b), not
  by whatever order the config happens to list them in.
* Samples inside a group are identity, not magnitude -> distinguished by MARKER
  SHAPE on one ink colour, never by six more hues. Six categorical hues cannot
  be kept colour-blind-safe, and the group colour is already carrying a job.
* Density is MAGNITUDE -> one hue, light to dark. Never a rainbow: on a rainbow
  the reader cannot tell which of two regions is higher without the colourbar.
* A signed group difference is POLARITY -> two hues around a near-white middle,
  blue for down and red for up.
* Grey is reserved, and split in two. `NO_RESULT` is a region the pipeline never
  produced a number for; `NOT_SIG` is a region that was tested and did not pass.
  Collapsing those into one grey is the single most common way these maps get
  misread, so they are drawn as two different greys with two legend entries.
"""
import os

import numpy as np

# --- categorical: groups -----------------------------------------------------
GROUP_COLORS = {"a": "#2a78d6", "b": "#eb6834"}      # blue, orange
GROUP_FILL = {"a": "#a9c9f2", "b": "#f5bda4"}        # same hues, bar fill

# Fixed hue order for the six base marker classes. The order is the
# colour-blind-safety mechanism, not decoration: it was validated on the
# adjacent-pair list (worst CVD dE 9.1, worst normal-vision dE 19.6), which is
# the list that matters for stacked bars. Three of the six sit below 3:1 against
# a white surface, so any chart using them must carry visible in-segment labels.
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

# --- identity: samples within a group ---------------------------------------
SAMPLE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdbd6"

# --- magnitude: sequential blue ramp ----------------------------------------
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_ORANGE = ["#fbe0d3", "#f5bda4", "#ef9a75", "#eb6834", "#c8501f", "#9d3d16", "#722c0f"]

# --- polarity: diverging blue <-> red ---------------------------------------
DIV_BLUE_RED = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb",
                "#f0efec",
                "#fbd5d4", "#f08b8a", "#d63b3a", "#8f1f1f"]

# --- reserved greys ----------------------------------------------------------
NO_RESULT = "#e8e7e3"   # never tested / no row in region_stats.csv
# Legend swatch for keys whose real fill comes from a colour scale, not from a
# fixed colour. Light enough to read as "a colour goes here", distinct from the
# NOT_SIG grey below, which IS a fixed meaning.
SWATCH_NEUTRAL = "#ece9e4"
NOT_SIG = "#c4c3be"     # tested, did not pass the correction
OUTSIDE = "#ffffff"     # outside the brain


def sequential_cmap(name="blue"):
    from matplotlib.colors import LinearSegmentedColormap
    steps = SEQ_BLUE if name == "blue" else SEQ_ORANGE
    return LinearSegmentedColormap.from_list(f"seq_{name}", steps)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div_blue_red", DIV_BLUE_RED)


def apply_rcparams():
    """Slide-legible defaults. Called by every script's main()."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK_SOFT,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",     # keep text editable in Illustrator
        "pdf.fonttype": 42,
    })


def sample_style(samples):
    """-> {sample: marker}. Fixed by position so a sample keeps its marker
    across every figure in the set."""
    return {s: SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)] for i, s in enumerate(samples)}


def p_label(p_adj, p_raw=None, alpha=0.05):
    """Text to put over a comparison.

    Deliberately NOT stars. With n=3 vs 3 a star invites the reader to treat the
    bar as a finding; the adjusted p and the family it came from are what decide
    that, so both are printed and the reader can see when it is 'n.s.'."""
    if p_adj is None or (isinstance(p_adj, float) and np.isnan(p_adj)):
        return "not tested"
    if p_adj < alpha:
        return f"p_adj = {p_adj:.3g}"
    if p_raw is not None and not np.isnan(p_raw) and p_raw < alpha:
        return f"n.s. (raw p = {p_raw:.3g})"
    return "n.s."


# Where each kind of figure lands under <output.dir>/figures/. A flat directory
# stops being usable somewhere around thirty files, and the full-cross run makes
# a hundred and fifty; the density maps in particular are one per class per
# level and swamp everything else, so they get their own.
SUBDIR = {
    "index": "00_index",
    "bars": "bars",
    "density": "density",
    "significance": "significance",
    "volcano": "volcano",
    "forest": "forest",
}


def figure_dir(out_dir, kind):
    """-> <out_dir>/<subdir for kind>, created. `out_dir` is the run's
    figures/ directory; an unknown kind falls back to it unchanged."""
    import os
    path = os.path.join(out_dir, SUBDIR[kind]) if kind in SUBDIR else out_dir
    os.makedirs(path, exist_ok=True)
    return path


def savefig(fig, out_path, dpi=200, also_pdf=True):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    written = [out_path]
    if also_pdf and out_path.lower().endswith(".png"):
        pdf = out_path[:-4] + ".pdf"
        fig.savefig(pdf)
        written.append(pdf)
    import matplotlib.pyplot as plt
    plt.close(fig)
    for w in written:
        print(f"Wrote {w}")
    return written
