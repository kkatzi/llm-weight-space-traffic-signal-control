# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Unified figure style for the paper.

Typeface. TeX Gyre Heros, the sans-serif of the paper's headings and TikZ diagrams. It is registered from
TeX Live's OpenType files and embedded as Type 3 (Type 42 is unreliable for CFF-based OpenType fonts).

Colour. The Okabe-Ito palette, assigned by role and checked with the dataviz palette validator (light
surface: lightness band, chroma floor, colour-vision-deficiency and normal-vision separation all pass).
ORANGE and PURPLE are below 3:1 contrast, so they are always paired with a marker shape or a direct label.
    LLM        vermilion       LLM-written or LLM-edited networks (blind arm)
    LLM_LOOP   reddish purple  LLM, closed-loop arm (with simulator feedback)
    LEARNED    blue            PPO, behaviour cloning, mechanical transfer, fine-tuning
    RULE       bluish green    execution-time rules (hold rule)
    SEARCH     orange          black-box search (evolution strategy)
    NEUTRAL    grey            random initialisation and other de-emphasised references
Classical controllers (actuated, max-pressure, fixed-time) appear as thin grey reference lines labelled
directly, not as additional colours.

Size. Figures are drawn at the width at which they are printed and included without scaling, so 8-pt
labels print at 8 pt: FULL is the text width of the A4 layout (2.3 cm margins), HALF is one of two panels side
by side.

Legends. Never inside the data area: above the axes (legend_above), beside them (legend_right), or replaced
by direct labels (label_end, ref_line).
"""
import glob
import os

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FULL, HALF = 6.4, 3.12           # inches; A4 text width is 6.46 in

VERMILION, BLUE, GREEN, ORANGE, PURPLE, SKY = "#D55E00", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9"
INK, MUTED, GREY, LIGHT, SPINE, GRID = "#1A1A1A", "#5F6368", "#8C8C8C", "#C5CCD4", "#4D4D4D", "#E6E6E6"
LLM, LLM_LOOP, LEARNED, RULE, SEARCH, NEUTRAL = VERMILION, PURPLE, BLUE, GREEN, ORANGE, GREY

# names used by the figure scripts, mapped onto the roles above
RED, NAVY, OCHRE, TEAL, YELLOW, LIGHTBLUE = LLM, LEARNED, SEARCH, RULE, SEARCH, LIGHT

COND = {  # condition -> (colour, label)
    "random": (NEUTRAL, "random init"),
    "bc": (LEARNED, "behaviour-cloning init"),
    "llm": (LLM, "LLM-written init"),
    "mech": (LEARNED, "mechanical transfer"),
    "llm_xfer": (LLM, "LLM transfer"),
    "finetune": (LEARNED, "PPO fine-tuning"),
    "patch_blind": (LLM, "LLM patch, blind"),
    "patch_loop": (LLM_LOOP, "LLM patch, 10 rounds"),
    "actuated": (MUTED, "tuned actuated"),
    "maxpressure": (MUTED, "max-pressure"),
    "fixed": (MUTED, "fixed-time"),
    "wrapper": (RULE, "hold rule at execution"),
    "es": (SEARCH, "(1+1)-ES, same budget"),
    "ppo_halt": (LEARNED, "PPO, halting reward"),
    "ppo_tts": (SKY, "PPO, TTS reward"),
}
REF_STYLE = {"actuated": (0, (4, 2)), "maxpressure": (0, (1, 1.5)), "fixed": (0, (4, 1.5, 1, 1.5))}

_FONT_DIRS = ["/usr/share/texmf/fonts/opentype/public/tex-gyre",
              os.path.expanduser("~/texlive/current/texmf-dist/fonts/opentype/public/tex-gyre"),
              "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre"]


def _register_font():
    for d in _FONT_DIRS:
        files = glob.glob(os.path.join(d, "texgyreheros-*.otf"))
        if files:
            for f in files:
                fm.fontManager.addfont(f)
            return "TeX Gyre Heros"
    return "DejaVu Sans"


def apply():
    family = _register_font()
    plt.rcParams.update({
        "font.family": family, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.title_fontsize": 7,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.sf": family,
        "axes.edgecolor": SPINE, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.axisbelow": True,
        "xtick.color": SPINE, "ytick.color": SPINE, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.3, "lines.markersize": 4, "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.2, "legend.labelspacing": 0.35,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 3, "svg.fonttype": "none",
    })


def finish(ax, grid=None):
    """Spines and grid. grid: None, 'x' or 'y' (a faint grid along that axis only)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.6)
    ax.grid(False)
    if grid in ("x", "y"):
        getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.5)


def legend_above(ax, ncol=2, handles=None, labels=None, fig=None, y=1.02):
    """A frameless legend above the axes (or above the whole figure if fig is given), outside the data."""
    kw = dict(loc="lower center", ncol=ncol, frameon=False, borderaxespad=0.0)
    if fig is not None:
        h, l = (handles, labels) if handles is not None else ax.get_legend_handles_labels()
        return fig.legend(h, l, bbox_to_anchor=(0.5, y), **kw)
    if handles is not None:
        return ax.legend(handles, labels, bbox_to_anchor=(0.5, y), **kw)
    return ax.legend(bbox_to_anchor=(0.5, y), **kw)


def legend_right(ax, handles=None, labels=None):
    """A frameless legend to the right of the axes, outside the data."""
    kw = dict(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, borderaxespad=0.0)
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


def label_end(ax, x, y, text, color, dx=3, dy=0, ha="left", **kw):
    """Direct label next to a data point (in place of a legend entry)."""
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                       fontsize=6.8, color=color, **kw)


def ref_line(ax, y, key, label=None, side="right", inside=False):
    """A classical controller as a thin grey reference line, labelled at the edge of the axes (outside), or
    just above the line at its right end (inside=True, for panels with a neighbour on the right)."""
    ax.axhline(y, color=MUTED, lw=0.8, ls=REF_STYLE.get(key, (0, (4, 2))), zorder=1)
    if inside:
        ax.annotate(label or COND[key][1], xy=(0.99, y), xycoords=("axes fraction", "data"), xytext=(0, 1.5),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=MUTED)
        return
    ax.annotate(label or COND[key][1], xy=(1.0 if side == "right" else 0.0, y),
                xycoords=("axes fraction", "data"), xytext=(3 if side == "right" else -3, 0),
                textcoords="offset points", ha="left" if side == "right" else "right", va="center",
                fontsize=6.5, color=MUTED, annotation_clip=False)


def panel(ax, letter, text="", pad=4):
    """Panel title: a bold letter followed by a short plain title, left-aligned above the axes."""
    ax.set_title(rf"$\mathbf{{{letter}}}$" + ("  " + text if text else ""), loc="left", pad=pad)


def save(fig, path, width=None):
    """Save with a tight bounding box at an exact printed width (inches). Text and markers keep their point
    size; only the axes stretch. The figure width is adjusted until the tight extent equals `width`."""
    if width is not None:
        pad = plt.rcParams["savefig.pad_inches"]
        for _ in range(6):
            r = fig.canvas.get_renderer()
            w = fig.get_tightbbox(r).width + 2 * pad
            if abs(w - width) < 0.01:
                break
            fig.set_size_inches(max(fig.get_figwidth() + (width - w), 0.5), fig.get_figheight())
    fig.savefig(path)
