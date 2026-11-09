# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Paper figures and tables from the study results -> paper/figs/*.pdf, paper/tables/*.tex (or $WSP_PAPER_DIR).

    (from the repository root)  python -m wsp.figures

Inputs: results/paper_numbers.json (wsp.analysis), results/secondary.json (wsp.secondary), the cached test
evaluations (wsp.stats.load), each training run's curve_val.json and monitor.csv. Style: wsp/plot_style.py
(TeX Gyre Heros, Okabe-Ito colours by role, legends outside the data, drawn at printed width).
"""
import json
import os
import sys

import numpy as np

from wsp.evaluate import V3, SETS
from wsp.stats import load, paired, pooled

PAPER = os.environ.get("WSP_PAPER_DIR", os.path.join(V3, "paper"))
FIGS, TABS = os.path.join(PAPER, "figs"), os.path.join(PAPER, "tables")
NETS = (0, 1, 2)
SC = ("low", "nominal", "high")
SL = {"low": slice(0, 10), "nominal": slice(10, 20), "high": slice(20, 30)}
O = json.load(open(os.path.join(V3, "results", "paper_numbers.json")))
SEC = json.load(open(os.path.join(V3, "results", "secondary.json")))
L2W_SCRATCH = ["ppo_s200", "ppo_s201", "ppo_s202", "ppo_s203_r1", "ppo_s204"]


def minus(s):
    return s.replace("-", "$-$") if "$" not in s else s


def pct(v, d=1, sign=True):
    if round(v, d) == 0:
        return f"{0:.{d}f}"
    return minus(f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def ci_txt(r, d=1):
    return f"{pct(r['pct'], d)} [{pct(r['pct_lo'], d)}, {pct(r['pct_hi'], d)}]"


def star(r, d=1):
    return pct(r["pct"], d) + ("$^{*}$" if r["sig"] else "")


def curve(problem, run):
    p = os.path.join(V3, "models", problem, run, "curve_val.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))
    return np.array(c["steps"]), np.array(c["tts"])


def valref(problem, label):
    d = json.load(open(os.path.join(V3, "results", "eval", problem, label, "val.json")))
    return d["summary"]["all"]["tts_veh_h"]


def two_refs(ax, S, problem, lower_below=False):
    """Tuned actuated and fixed-time validation levels, labelled at the right end: above the line, or for the lower
    line below it (lower_below=True, where the curves run well below both lines)."""
    refs = sorted([(valref(problem, "actuated"), "actuated", "tuned actuated"),
                   (valref(problem, "fixed"), "fixed", "tuned fixed time")], reverse=True)
    for i, (v, key, lab) in enumerate(refs):
        ax.axhline(v, color=S.MUTED, lw=0.8, ls=S.REF_STYLE[key], zorder=1)
        ax.annotate(lab, xy=(0.99, v), xycoords=("axes fraction", "data"), xytext=(0, -1.5 if (i == 1 and lower_below) else 1.5),
                    textcoords="offset points", ha="right", va="top" if (i == 1 and lower_below) else "bottom", fontsize=6.5,
                    color=S.MUTED)


def monitor(problem, run):
    rows = [l.split(",") for l in open(os.path.join(V3, "models", problem, run, "monitor.csv")).read().splitlines()[2:]
            if l.strip()]
    R = np.array([float(r[0]) for r in rows])
    return -50.0 * R / 3600.0          # total delay per training episode (veh-h)


def write_table(name, lines):
    os.makedirs(TABS, exist_ok=True)
    open(os.path.join(TABS, name), "w").write("\n".join(lines) + "\n")


# =============================================================== figures
def fig_demand(S, plt):
    from wsp import junctions as J
    MAIN_COL, CROSS_COL = "#3D4A57", "#B9C3CE"
    mb, cb = J.JUNCTIONS["L2"]["base"]
    fig = plt.figure(figsize=(S.FULL, 3.6))
    gs = fig.add_gridspec(2, 6, hspace=0.8, wspace=1.1, top=0.86)
    for i, name in enumerate(SC):
        sc = J.scenario("L2", name)
        prof = np.array(sc["profile"]) * sc["scale"]
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        x = np.arange(4)
        ax.bar(x - 0.18, mb * prof, width=0.34, color=MAIN_COL, label="main road (each east or west approach)")
        ax.bar(x + 0.18, cb * prof, width=0.34, color=CROSS_COL, label="cross street (each north or south approach)")
        ax.set_xticks(x)
        ax.set_xticklabels(["0–15", "15–30", "30–45", "45–60"])
        ax.set_ylim(0, 1000)
        ax.set_xlabel("minutes into the hour")
        if i == 0:
            ax.set_ylabel("arrivals per approach (veh/h)")
        else:
            ax.set_yticklabels([])
        S.finish(ax, grid="y")
        S.panel(ax, "abc"[i], f"Test: {name} (×{sc['scale']:g})")
    S.legend_above(fig.axes[0], ncol=2, fig=fig, y=0.995)
    rng = np.random.default_rng(0)
    tot, share = [], []
    for _ in range(5000):
        sc = J.sample_train("L2", rng)
        m = mb * np.exp(sc["bias"]) * sc["scale"] * np.array(sc["profile"])
        c = cb * np.exp(-sc["bias"]) * sc["scale"] * np.array(sc["profile"])
        tot.append(2 * m.mean() + 2 * c.mean())
        share.append(100 * m.mean() / (m.mean() + c.mean()))
    tests = {name: (2 * (mb + cb) * J.SCENARIO_SCALE[name] * np.mean(J.NOMINAL_PROFILE), 100 * mb / (mb + cb))
             for name in SC}
    for j, (vals, xlabel, rx, title) in enumerate([
            (tot, "total traffic entering the junction (veh/h)", (1000, 4000), "Training range: amount of traffic"),
            (share, "share of the traffic on the main road (%)", (40, 80), "Training range: which road is busier")]):
        ax = fig.add_subplot(gs[1, 3 * j:3 * j + 3])
        ax.hist(vals, bins=40, range=rx, color=CROSS_COL, lw=0)
        top = ax.get_ylim()[1]
        if j == 0:
            for n_, name in enumerate(SC):
                v = tests[name][0]
                ax.axvline(v, color=S.INK, lw=0.9)
                dx = (rx[1] - rx[0]) * 0.012
                ax.text(v - dx if n_ < 2 else v + dx, top * 1.08, name, fontsize=6.5, color=S.INK,
                        ha="right" if n_ < 2 else "left", va="bottom")
        else:
            v = tests["nominal"][1]
            ax.axvline(v, color=S.INK, lw=0.9)
            ax.text(v + 0.5, top * 1.08, "all test scenarios", fontsize=6.5, color=S.INK, ha="left", va="bottom")
        S.panel(ax, "de"[j], title, pad=10)
        ax.set_xlim(*rx)
        ax.set_ylim(0, top * 1.25)
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        S.finish(ax)
        ax.spines["left"].set_visible(False)
    S.save(fig, os.path.join(FIGS, "demand.pdf"), S.FULL)
    plt.close(fig)


def fig_reference(S, plt):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.2), sharey=True)
    for ax, (prob, runs, letter, title) in zip(axes, [
            ("L2", [f"ppo_s{s}" for s in range(100, 105)], "a", "Junction L2 (the agents' targets)"),
            ("L2W", L2W_SCRATCH, "b", "Junction L2W (Transfer target)")]):
        act, fx = load(prob, "actuated")["tts"], load(prob, "fixed")["tts"]
        cats = list(SC) + ["all"]
        for i, c in enumerate(cats):
            sl = SL.get(c, slice(0, 30))
            ref = act[sl].mean()
            ax.scatter([i - 0.15], [100 * (fx[sl].mean() / ref - 1)], marker="D", s=18, color=S.MUTED,
                       edgecolor="white", lw=0.5, zorder=4)
            ys = [100 * (load(prob, r)["tts"][sl].mean() / ref - 1) for r in runs]
            ax.scatter([i + 0.12] * len(ys), ys, s=14, color=S.LEARNED, edgecolor="white", lw=0.4, zorder=4)
        ax.axhline(0, color=S.MUTED, lw=0.8, ls=S.REF_STYLE["actuated"], zorder=1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(["low", "nominal", "high", "all 30"])
        ax.set_xlim(-0.6, len(cats) - 0.4)
        ax.set_xlabel("test scenario")
        if letter == "a":
            ax.set_ylabel("test TTS vs. tuned actuated (%)")
        S.finish(ax, grid="y")
        S.panel(ax, letter, title, pad=4)
    h = [Line2D([], [], color=S.LEARNED, marker="o", ls="", ms=4.5, mec="white", mew=0.5, label="PPO (5 seeds)"),
         Line2D([], [], color=S.MUTED, marker="D", ls="", ms=4.5, mec="white", mew=0.5, label="tuned fixed time"),
         Line2D([], [], color=S.MUTED, lw=0.8, ls=S.REF_STYLE["actuated"], label="tuned actuated (= 0)")]
    S.legend_above(axes[0], ncol=3, handles=h, labels=[x.get_label() for x in h], fig=fig, y=1.0)
    fig.subplots_adjust(top=0.82)
    S.save(fig, os.path.join(FIGS, "reference.pdf"), S.FULL)
    plt.close(fig)


def _band(ax, S, problem, runs, color, label, xmax, ls="-"):
    cs = [curve(problem, r) for r in runs]
    cs = [c for c in cs if c is not None]
    steps = sorted(set.intersection(*[set(c[0][c[0] <= xmax].tolist()) for c in cs]))
    Y = np.array([[c[1][list(c[0]).index(s)] for s in steps] for c in cs])
    x = np.array(steps) / 1e6
    ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.14, lw=0)
    ax.plot(x, np.median(Y, 0), color=color, lw=1.3, ls=ls, label=f"{label} ({len(cs)} runs)")


def fig_curves(S, plt):
    # PPO references: every seed, both junctions
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.25))
    for ax, (prob, runs, letter, title, ylim) in zip(axes, [
            ("L2", [f"ppo_s{s}" for s in range(100, 105)], "a", "L2, PPO from scratch", (44, 60)),
            ("L2W", L2W_SCRATCH, "b", "L2W, PPO from scratch", (58, 80))]):
        for r in runs:
            st, y = curve(prob, r)
            ax.plot(st / 1e6, y, color=S.LEARNED, lw=0.9, alpha=0.85)
        two_refs(ax, S, prob, lower_below=prob == "L2W")
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 4.8 if prob == "L2W" else 3)
        ax.set_xlabel("PPO training steps (millions)")
        if letter == "a":
            ax.set_ylabel("validation TTS (veh-h)")
        S.finish(ax, grid="y")
        S.panel(ax, letter, title + " (one line per seed)")
    S.save(fig, os.path.join(FIGS, "ppo_curves.pdf"), S.FULL)
    plt.close(fig)

    # Initialise and Transfer: validation curves + training-episode delay
    for name, prob, warm, scratch, lab, ylim, ycost in [
            ("init", "L2", [f"init_r{k}_s{s}" for k in (1, 2, 3) for s in (400, 401)],
             [f"ppo_s{s}" for s in range(100, 105)], "PPO from the LLM-written network", (44, 60), 160),
            ("transfer", "L2W", [f"transfer_net{k}_s{s}" for k in NETS for s in (500, 501)], L2W_SCRATCH,
             "PPO from the LLM transfer", (58, 80), 300)]:
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.FULL, 2.4))
        _band(ax, S, prob, scratch, S.NEUTRAL, "PPO from random initialisation", 3_000_000)
        _band(ax, S, prob, warm, S.LLM, lab, 3_000_000)
        two_refs(ax, S, prob, lower_below=prob == "L2W")
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 3)
        ax.set_xlabel("PPO training steps (millions)")
        ax.set_ylabel("validation TTS (veh-h)")
        S.finish(ax, grid="y")
        S.panel(ax, "a", "Learning curves", pad=4)
        n = 200
        for runs, col in ((scratch, S.NEUTRAL), (warm, S.LLM)):
            D = np.array([monitor(prob, r)[:n] for r in runs])
            k = 10
            sm = np.array([np.convolve(d, np.ones(k) / k, mode="valid") for d in D])
            x = np.arange(k, n + 1)
            bx.fill_between(x, sm.min(0), sm.max(0), color=col, alpha=0.14, lw=0)
            bx.plot(x, np.median(sm, 0), color=col, lw=1.3)
        thr = O["initialise" if name == "init" else "transfer"]["threshold_unacceptable"]
        S.ref_line(bx, thr, "fixed", "fixed time, 95th percentile", inside=True)
        bx.set_xlim(0, n)
        bx.set_ylim(0, ycost)
        bx.set_xlabel("training episode")
        bx.set_ylabel("delay in the episode (veh-h)")
        S.finish(bx, grid="y")
        S.panel(bx, "b", "Cost of learning (10-episode mean)", pad=4)
        S.legend_above(ax, ncol=2, fig=fig, y=1.0)
        fig.subplots_adjust(top=0.8, wspace=0.3)
        S.save(fig, os.path.join(FIGS, f"{name}_curves.pdf"), S.FULL)
        plt.close(fig)


def _dot_rows(ax, S, groups, xlabel):
    """groups: [(label, colour, [paired-result dicts])] -> one row per run with its 95 % interval."""
    y, centres, names = 0.0, [], []
    for label, col, rows in groups:
        ys = []
        for r in rows:
            ax.plot([r["pct_lo"], r["pct_hi"]], [y, y], color=col, lw=1.1, solid_capstyle="butt")
            ax.scatter([r["pct"]], [y], s=14, color=col, edgecolor="white", lw=0.5, zorder=4)
            ys.append(y)
            y -= 0.8
        centres.append(np.mean(ys))
        names.append(label)
        y -= 0.7
    ax.axvline(0, color=S.MUTED, lw=0.6)
    ax.set_yticks(centres)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    S.finish(ax, grid="x")
    ax.spines["left"].set_visible(False)


def fig_patch(S, plt):
    P = O["patch"]
    per = lambda fmt: [P["runs"][fmt.format(k=k)]["vs_target"] for k in NETS]  # noqa: E731
    groups = [("Opus, one shot", S.LLM, per("net{k}_b0")), ("Opus, 5 calls", S.LLM_LOOP, per("net{k}_b5")),
              ("Opus, 10 calls", S.LLM_LOOP, per("net{k}_b10"))]
    for m, nm in (("fable", "Fable"), ("sonnet", "Sonnet"), ("haiku", "Haiku")):
        groups.append((f"{nm}, 10 calls", S.LLM_LOOP, [P["writers"][m]["runs"][str(k)]["vs_target"] for k in NETS]))
    for key, nm, col in (("es_seed0", "(1+1)-ES, seed 0", S.SEARCH), ("es_seed1", "(1+1)-ES, seed 1", S.SEARCH),
                         ("wrapper", "Hold rule at execution", S.RULE),
                         ("finetune_300k", "PPO fine-tuning, 300k", S.LEARNED),
                         ("finetune_1M", "PPO fine-tuning, 1M", S.LEARNED)):
        groups.append((nm, col, [P["baselines"][key]["per_target"][str(k)] for k in NETS]))
    fig, ax = plt.subplots(figsize=(0.8 * S.FULL, 3.6))
    _dot_rows(ax, S, groups, "change in test TTS vs. the target network (%), 95 % interval")
    S.save(fig, os.path.join(FIGS, "patch.pdf"), 0.8 * S.FULL)
    plt.close(fig)


def _episodes():
    b = SEC["cost"]["baselines"]
    ft = {k: v for k, v in b["ft_fault"].items() if v}
    return {"ft300": np.mean([v["episodes"] for k, v in ft.items() if "@" in k]),
            "ft1M": np.mean([v["episodes"] for k, v in ft.items() if "@" not in k]), "es": b["es"]["episodes_per_run"]}


def fig_fault(S, plt):
    from matplotlib.lines import Line2D
    F = O["patch_fault"]
    ep = _episodes()
    calls = {f"b{b}": {k: F["methods"][f"agents_b{b}"]["runs"][str(k)]["calls_used"] for k in NETS} for b in (0, 5, 10)}
    pts = [("agents_b0", lambda k: 0, S.LLM, "o", "Opus, one shot"),
           ("agents_b5", lambda k: 9 * calls["b5"][k], S.LLM_LOOP, "v", "Opus, 5 calls"),
           ("agents_b10", lambda k: 9 * calls["b10"][k], S.LLM_LOOP, "o", "Opus, 10 calls"),
           ("mean_imputation", lambda k: 0, S.NEUTRAL, "s", "mean imputation"),
           ("es_seed0", lambda k: ep["es"], S.SEARCH, "^", "(1+1)-ES, 10 evaluations"),
           ("es_seed1", lambda k: ep["es"], S.SEARCH, "^", None),
           ("finetune_300k", lambda k: ep["ft300"], S.LEARNED, "D", "PPO fine-tuning, 300k"),
           ("finetune_1M", lambda k: ep["ft1M"], S.LEARNED, "s", "PPO fine-tuning, 1M")]
    fig, axes = plt.subplots(1, 3, figsize=(S.FULL, 2.35), sharey=True)
    for k, ax in zip(NETS, axes):
        for key, xf, col, mk, lab in pts:
            v = F["methods"][key]["runs"][str(k)]["recovered_pct"]
            jitter = {"agents_b0": -0.6, "mean_imputation": 0.6}.get(key, 0)
            ax.scatter([xf(k) + jitter], [v], color=col, marker=mk, s=24, edgecolor="white", lw=0.5, zorder=5)
        R = O.get("patch_fault_rep")
        if R:
            for b_, col, mk in ((0, S.LLM, "o"), (10, S.LLM_LOOP, "o")):
                for r in R["arms"][f"b{b_}"]["sessions"][str(k)][1:]:
                    x = 9 * r["calls_used"] + (-0.6 if b_ == 0 else 0)
                    v = r["recovered_pct"]
                    if v < -5:          # off the scale (a gridlocked test episode): edge marker with its value
                        ax.scatter([x], [-3], facecolor="white", edgecolor=col, marker="v", s=26, lw=1.0, zorder=4,
                                   clip_on=False)
                        ax.annotate(f"\u2212{abs(v):.0f} %: one test\nepisode gridlocks", (x, -3), xytext=(-6, 7),
                                    textcoords="offset points", fontsize=6.3, color=S.MUTED, va="bottom", ha="right")
                        continue
                    ax.scatter([x], [v], facecolor="white", edgecolor=col, marker=mk, s=20, lw=1.0, zorder=4)
        ax.axhline(100, color=S.MUTED, lw=0.6)
        ax.axhline(0, color=S.MUTED, lw=0.6)
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(-2, 3000)
        ax.set_xticks([0, 10, 100, 1000])
        ax.set_xticklabels(["0", "10", "100", "1,000"])
        ax.set_ylim(-5, 108)
        ax.set_xlabel("simulated episodes used")
        if k == 0:
            ax.set_ylabel("share of the fault's cost recovered (%)")
        S.finish(ax, grid="y")
        h, f_ = F["healthy"][str(k)]["all"], F["faulty"][str(k)]["all"]
        S.panel(ax, "abc"[k], f"Net {k}: faulty {f_:.1f}, healthy {h:.1f}", pad=4)
    hs = [Line2D([], [], color=c, marker=m, ls="", ms=5, mec="white", mew=0.5, label=l) for _, _, c, m, l in pts if l]
    if O.get("patch_fault_rep"):
        hs.append(Line2D([], [], color=S.MUTED, marker="o", ls="", ms=4.5, mfc="white", mew=1.0,
                         label="replicate sessions (hollow)"))
    S.legend_above(axes[0], ncol=4, handles=hs, labels=[x.get_label() for x in hs], fig=fig, y=1.0)
    fig.subplots_adjust(top=0.77, wspace=0.12)
    S.save(fig, os.path.join(FIGS, "patch_fault.pdf"), S.FULL)
    plt.close(fig)


def fig_explain(S, plt):
    E = O["explain"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.FULL, 2.35))
    style = {"agent": (S.LLM, "o", "LLM rule files"), "cart": (S.LEARNED, "s", "decision trees (CART)"),
             "dagger": (S.NEUTRAL, "^", "iterative trees (DAgger)")}
    CLIP = 25
    for fam, (col, mk, lab) in style.items():
        xs, fs, ts = [], [], []
        if fam == "agent":
            for k in NETS:
                for n in ("small", "full"):
                    a = E["agents"][str(k)][n]
                    xs.append(max(a["comparisons"], 1)), fs.append(100 * a["fidelity"]), ts.append(a["vs_network"]["pct"])
        else:
            for lab_, t in E["trees"].items():
                if lab_.startswith(fam):
                    xs.append(t["comparisons"]), fs.append(100 * t["fidelity"]), ts.append(t["vs_network"]["pct"])
        ax.scatter(xs, fs, color=col, marker=mk, s=20, edgecolor="white", lw=0.5, zorder=4, label=lab)
        tc = [min(t, CLIP) for t in ts]
        bx.scatter(xs, tc, color=col, marker=mk, s=20, edgecolor="white", lw=0.5, zorder=4)
        for x, t in zip(xs, ts):
            if t > CLIP:
                bx.annotate(f"+{t:.0f} %", (x, CLIP), xytext=(4, -1), textcoords="offset points", fontsize=6.5,
                            color=S.MUTED, va="center")
    for a in (ax, bx):
        a.set_xscale("log")
        a.set_xlabel("comparisons in the rule file")
        S.finish(a, grid="y")
    ax.set_ylabel("held-out fidelity (%)")
    ax.set_ylim(35, 100)
    bx.axhline(0, color=S.MUTED, lw=0.6)
    bx.set_ylim(-2, CLIP + 2)
    bx.set_ylabel("test TTS vs. the network (%)")
    S.panel(ax, "a", "Agreement with the network", pad=4)
    S.panel(bx, "b", "Closed loop, as the controller", pad=4)
    S.legend_above(ax, ncol=3, fig=fig, y=1.0)
    fig.subplots_adjust(top=0.8, wspace=0.3)
    S.save(fig, os.path.join(FIGS, "explain.pdf"), S.FULL)
    plt.close(fig)


def fig_constrain(S, plt):
    C = O["constrain"]
    N = {k: load("L2_rules", f"naive_net{k}")["tts"] for k in NETS}
    Nm = np.mean([N[k] for k in NETS], axis=0)
    rows = lambda fmt: [paired(load("L2_rules", fmt.format(k=k))["tts"], N[k]) for k in NETS]  # noqa: E731
    groups = [("LLM supervisor, one shot", S.LLM, [C["runs"][f"net{k}_b0"]["vs_naive"] for k in NETS]),
              ("LLM supervisor, 10 calls", S.LLM_LOOP, [C["runs"][f"net{k}_b10"]["vs_naive"] for k in NETS]),
              ("Network, best legal phase", S.NEUTRAL, rows("masked_net{k}")),
              ("PPO fine-tuned under the rules, 300k", S.LEARNED, rows("ft_net{k}_s70{k}_300k")),
              ("PPO fine-tuned under the rules, 1M", S.LEARNED, rows("ft_net{k}_s70{k}_1000k")),
              ("PPO trained under the rules, 3M", S.LEARNED,
               [paired(load("L2_rules", f"ppo_s{s}")["tts"], Nm) for s in range(300, 303)])]
    fig, ax = plt.subplots(figsize=(0.8 * S.FULL, 2.6))
    _dot_rows(ax, S, groups, "change in test TTS vs. network + shield (%), 95 % interval")
    S.save(fig, os.path.join(FIGS, "constrain.pdf"), 0.8 * S.FULL)
    plt.close(fig)


def fig_truncation(S, plt):
    T = SEC["truncation"]
    fig, axes = plt.subplots(1, 2, figsize=(S.FULL, 2.3))
    for ax, case, ylab, key in ((axes[0], "patch", "test TTS vs. target (%)", "vs_ref"),
                                (axes[1], "patch_fault", "fault's cost recovered (%)", "recovered_pct")):
        for k, mk in zip(NETS, "os^"):
            for b, col in ((5, S.LLM_LOOP), (10, S.LLM_LOOP)):
                t = T.get(f"{case}/net{k}_b{b}")
                if not t:
                    continue
                xs = [s["after_call"] for s in t["steps"] if key in s or "vs_ref" in s]
                ys = [(s[key]["pct"] if key == "vs_ref" else s[key]) for s in t["steps"] if (key in s)]
                ax.plot(xs[:len(ys)], ys, color=col, lw=0.9, ls="-" if b == 10 else (0, (3, 1.5)), marker=mk, ms=3.5,
                        mec="white", mew=0.4, alpha=0.9)
        ax.set_xlabel("simulator calls made")
        ax.set_ylabel(ylab)
        ax.set_xticks(range(1, 11))
        S.finish(ax, grid="y")
    axes[0].axhline(0, color=S.MUTED, lw=0.6)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=S.LLM_LOOP, marker=m, ls="", ms=4, mec="white", mew=0.4, label=f"Net {k}")
          for k, m in zip(NETS, "os^")]
    hs += [Line2D([], [], color=S.LLM_LOOP, lw=0.9, ls="-", label="10-call run"),
           Line2D([], [], color=S.LLM_LOOP, lw=0.9, ls=(0, (3, 1.5)), label="5-call run")]
    S.legend_above(axes[0], ncol=5, handles=hs, labels=[h.get_label() for h in hs], fig=fig, y=1.0)
    fig.subplots_adjust(top=0.8)
    S.panel(axes[0], "a", "Patch, converged targets", pad=4)
    S.panel(axes[1], "b", "Patch, failed detector", pad=4)
    S.save(fig, os.path.join(FIGS, "truncation.pdf"), S.FULL)
    plt.close(fig)


# =============================================================== tables
def tab_reference():
    R = O["reference"]
    L = ["\\begin{tabular}{llrrrr}", "\\toprule",
         "Junction & Controller & Low & Nominal & High & Mean \\\\", "\\midrule"]
    for prob, name, runs in (("L2", "L2", [f"ppo_s{s}" for s in range(100, 105)]), ("L2W", "L2W", L2W_SCRATCH),
                             ("L2_rules", "L2 + rules", [f"ppo_s{s}" for s in range(300, 303)])):
        first = True
        for lab, txt in (("fixed", "Fixed time (tuned)"), ("actuated", "Actuated (tuned)")):
            x = load(prob, lab)["tts"]
            L.append(f"{name if first else ''} & {txt} & " + " & ".join(f"{x[SL[s]].mean():.1f}" for s in SC)
                     + f" & {x.mean():.1f} \\\\")
            first = False
        X = np.array([load(prob, r)["tts"] for r in runs])
        means = X.mean(1)
        L.append(f" & PPO ({len(runs)} seeds, mean) & " + " & ".join(f"{X[:, SL[s]].mean():.1f}" for s in SC)
                 + f" & {X.mean():.1f} ({means.min():.1f}--{means.max():.1f}) \\\\")
        r = R[prob]
        L.append(f" & PPO vs.\\ actuated (\\%) & \\multicolumn{{4}}{{r}}{{{ci_txt(r['ppo_vs_actuated'])}}} \\\\")
        L.append(f" & PPO vs.\\ fixed time (\\%) & \\multicolumn{{4}}{{r}}{{{ci_txt(r['ppo_vs_fixed'])}}} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    write_table("reference.tex", L)
    # tuned parameters (appendix)
    L = ["\\begin{tabular}{lll}", "\\toprule", "Junction & Fixed time: greens (s) & Actuated: extension, max.\\ green, detectors \\\\",
         "\\midrule"]
    for prob, name in (("L2", "L2"), ("L2W", "L2W"), ("L2_rules", "L2 + rules")):
        t = R[prob]["tuned"]
        g = t["fixed"].split(":")[1].replace(",", ", ")
        e, m, lay = t["actuated"].split(":")[1].split(",")
        L.append(f"{name} & {g} & {float(e):g} veh, {float(m):g} s, {'stop-line' if lay == 'stop' else 'stop-line + upstream'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("tuned.tex", L)


def tab_warm():
    for name, d_key, prob, zero_lab in (("init", "initialise", "L2", "LLM-written network {k}"),
                                        ("transfer", "transfer", "L2W", "LLM transfer of Net {k}")):
        d = O[d_key]
        L = ["\\begin{tabular}{lrrrr}", "\\toprule"]
        hdr = "Network (no training) & Test TTS & vs.\\ fixed time & vs.\\ actuated" + (" & vs.\\ padding" if name == "transfer" else "")
        L[0] = "\\begin{tabular}{l" + "r" * (4 if name == "transfer" else 3) + "}"
        L.append(hdr + " \\\\")
        L.append("\\midrule")
        for k, z in d["zero_shot"].items():
            row = f"{zero_lab.format(k=k)} & {z['tts']['all']:.1f} & {star(z['vs_fixed'])} & {star(z['vs_actuated'])}"
            if name == "transfer":
                row += f" & {star(z['vs_pad'])}"
            L.append(row + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        write_table(f"{name}_zero.tex", L)
        # cost of learning
        L = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Training runs & Delay, first 50 ep.\\ (veh-h) & Delay, first 1M steps (veh-h) & Episodes above threshold, first 200 & Final test TTS \\\\",
             "\\midrule"]
        for grp, lab in (("scratch", "PPO from random initialisation"),
                         ("warm", "PPO from the LLM network" if name == "init" else "PPO from the LLM transfer")):
            rows = list(d[grp].values())
            f50 = [r["first50_mean_delay"] for r in rows]
            f1m = [r["first1M_total_delay"] for r in rows]
            un = [r["unacceptable_first200"] for r in rows]
            ft = [r["test"]["all"] for r in rows]
            L.append(f"{lab} ({len(rows)}) & {np.mean(f50):.1f} ({min(f50):.0f}--{max(f50):.0f}) & "
                     f"{np.mean(f1m):,.0f} & {np.mean(un):.1f} ({min(un)}--{max(un)}) & {np.mean(ft):.1f} \\\\".replace(",", "{,}"))
        L.append("\\midrule")
        L.append(f"Warm vs.\\ scratch (\\%, Welch) & {ci_txt(d['warm_vs_scratch_first50_mean_delay'])} & "
                 f"{ci_txt(d['warm_vs_scratch_first1M_total_delay'])} & & {ci_txt(d['warm_vs_scratch_final_tts'])} \\\\")
        L += ["\\bottomrule", "\\end{tabular}"]
        write_table(f"{name}_cost.tex", L)


def tab_patch():
    P = O["patch"]
    ep = SEC["cost"]["baselines"]["ft_patch"]
    e300 = np.mean([v["episodes"] for k, v in ep.items() if v and "@" in k])
    e1m = np.mean([v["episodes"] for k, v in ep.items() if v and "@" not in k])
    L = ["\\begin{tabular}{lrrrr}", "\\toprule", "Method (simulated episodes) & Net 0 & Net 1 & Net 2 & Pooled \\\\", "\\midrule"]

    def row(lab, rs, pool):
        L.append(f"{lab} & " + " & ".join(star(r, 2) for r in rs) + f" & {ci_txt(pool, 2)} \\\\")
    for b, lab in ((0, "Opus, one shot (0)"), (5, "Opus, 5 calls ($\\le$45)"), (10, "Opus, 10 calls ($\\le$90)")):
        row(lab, [P["runs"][f"net{k}_b{b}"]["vs_target"] for k in NETS], P["pooled"][str(b)])
    for m, nm in (("fable", "Fable"), ("sonnet", "Sonnet"), ("haiku", "Haiku")):
        row(f"{nm}, 10 calls ($\\le$90)", [P["writers"][m]["runs"][str(k)]["vs_target"] for k in NETS], P["writers"][m]["pooled"])
    for key, lab in (("es_seed0", "(1+1)-ES, seed 0 (90)"), ("es_seed1", "(1+1)-ES, seed 1 (90)"),
                     ("wrapper", "Hold rule at execution, exploratory (90)"),
                     ("finetune_300k", f"PPO fine-tuning, 300k ({e300:,.0f})"), ("finetune_1M", f"PPO fine-tuning, 1M ({e1m:,.0f})")):
        b = P["baselines"][key]
        row(lab.replace(",", "{,}", 1) if "PPO" in lab and False else lab, [b["per_target"][str(k)] for k in NETS], b["pooled"])
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("patch.tex", L)
    # edits (appendix)
    E = SEC["edits"]
    L = ["\\begin{tabular}{llrrrrrrrr}", "\\toprule",
         "Writer, budget & Target & $W_1$ & $b_1$ & $W_2$ & $b_2$ & $W_3$ & $b_3$ & Total & Calls \\\\", "\\midrule"]
    for case, arms in (("patch", [("b0", "Opus, one shot"), ("b5", "Opus, 5 calls"), ("b10", "Opus, 10 calls")]),
                       ("patch_models", [("fable", "Fable, 10 calls"), ("sonnet", "Sonnet, 10 calls"), ("haiku", "Haiku, 10 calls")]),
                       ("patch_fault", [("b0", "Fault: Opus, one shot"), ("b5", "Fault: Opus, 5 calls"), ("b10", "Fault: Opus, 10 calls")])):
        for arm, lab in arms:
            for k in NETS:
                pid = f"net{k}_{arm}"
                e = E[f"{case}/{pid}"]
                calls = len([1 for _ in open(os.path.join(V3, "packs", case, pid, "eval_log.jsonl"))]) \
                    if os.path.exists(os.path.join(V3, "packs", case, pid, "eval_log.jsonl")) else 0
                L.append(f"{lab if k == 0 else ''} & Net {k} & " + " & ".join(str(e["layers"][l]["changed"]) for l in
                                                                          ("W1", "b1", "W2", "b2", "W3", "b3"))
                         + f" & {e['total_changed']} & {calls} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    write_table("patch_edits.tex", L)


def tab_fault():
    F = O["patch_fault"]
    ep = _episodes()
    L = ["\\begin{tabular}{lrrrr}", "\\toprule",
         "Method (simulated episodes) & Net 0 & Net 1 & Net 2 & Pooled vs.\\ faulty \\\\", "\\midrule",
         "Healthy network (TTS) & " + " & ".join(f"{F['healthy'][str(k)]['all']:.1f}" for k in NETS) + " & \\\\",
         "Faulty network (TTS) & " + " & ".join(f"{F['faulty'][str(k)]['all']:.1f}" for k in NETS) + " & \\\\",
         "\\midrule", "\\multicolumn{5}{l}{\\emph{Share of the fault's cost recovered (\\%); pooled change in test TTS vs.\\ the faulty network}} \\\\"]
    for key, lab in (("agents_b0", "Opus, one shot (0)"), ("agents_b5", "Opus, 5 calls ($\\le$45)"),
                     ("agents_b10", "Opus, 10 calls ($\\le$90)"), ("mean_imputation", "Mean imputation (0)"),
                     ("es_seed0", f"(1+1)-ES, seed 0 ({ep['es']})"), ("es_seed1", f"(1+1)-ES, seed 1 ({ep['es']})"),
                     ("finetune_300k", f"PPO fine-tuning, 300k ({ep['ft300']:.0f})"),
                     ("finetune_1M", f"PPO fine-tuning, 1M ({ep['ft1M']:,.0f})".replace("1,", "1{,}"))):
        m = F["methods"][key]
        L.append(f"\\quad {lab} & " + " & ".join(
            f"{m['runs'][str(k)]['recovered_pct']:.0f}" + ("$^{*}$" if m["runs"][str(k)]["vs_faulty"]["sig"] else "")
            for k in NETS) + f" & {ci_txt(m['pooled_vs_faulty'])} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{5}{l}{\\emph{Opus, 10 calls, against each baseline: change in test TTS (\\%)}} \\\\")
    for base, lab in (("mean_imputation", "vs.\\ mean imputation"), ("es_seed0", "vs.\\ (1+1)-ES, seed 0"),
                      ("es_seed1", "vs.\\ (1+1)-ES, seed 1"), ("finetune_1M", "vs.\\ PPO fine-tuning, 1M")):
        r = F[f"agents_b10_vs_{base}"]
        L.append(f"\\quad {lab} & " + " & ".join(star(r["per_target"][str(k)]) for k in NETS) + f" & {ci_txt(r['pooled'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("patch_fault.tex", L)


def tab_fault_rep():
    R = O.get("patch_fault_rep")
    if not R:
        return
    L = ["\\begin{tabular}{llrrrrr}", "\\toprule",
         "Arm & Network & Session 1 & Session 2 & Session 3 & Mean & Mean of sessions vs.\\ faulty (\\%) \\\\", "\\midrule"]
    for b, lab in (("b0", "One shot"), ("b10", "10 calls")):
        arm = R["arms"][b]
        for k in NETS:
            ss = arm["sessions"][str(k)]
            m = arm["mean_of_sessions"][str(k)]
            L.append(f"{lab if k == 0 else ''} & Net {k} & " + " & ".join(pct(r['recovered_pct'], 0, sign=False) for r in ss)
                     + f" & {pct(m['recovered_mean'], 0, sign=False)} & {ci_txt(m['vs_faulty'])} \\\\")
        L.append("\\midrule")
    L.append("\\multicolumn{7}{l}{\\emph{Mean of the three sessions against each baseline: change in test TTS (\\%), per network and pooled}} \\\\")
    for b, lab in (("b0", "One shot"), ("b10", "10 calls")):
        for base, bl in (("mean_imputation", "mean imputation"), ("es_seed0", "(1+1)-ES, seed 0"),
                         ("es_seed1", "(1+1)-ES, seed 1"), ("finetune_1M", "PPO fine-tuning, 1M")):
            r = R["arms"][b][f"vs_{base}"]
            L.append(f"{lab} & vs.\\ {bl} & " + " & ".join(star(r["per_target"][str(k)]) for k in NETS)
                     + f" & & {ci_txt(r['pooled'])} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    write_table("patch_fault_rep.tex", L)


def tab_explain():
    E = O["explain"]
    L = ["\\begin{tabular}{llrrrr}", "\\toprule",
         "Target & Rule file & Comparisons & Constants & Fidelity (\\%) & Test TTS vs.\\ network (\\%) \\\\", "\\midrule"]
    for k in NETS:
        rows = [(f"LLM, small", E["agents"][str(k)]["small"]), (f"LLM, full", E["agents"][str(k)]["full"])]
        for fam, nm in (("cart", "CART"), ("dagger", "DAgger")):
            for n, d in (("small", 3), ("mid", 5), ("full", 8)):
                rows.append((f"{nm}, depth {d}", E["trees"][f"{fam}_net{k}_{n}"]))
        for i, (nm, r) in enumerate(rows):
            L.append(f"{'Net ' + str(k) if i == 0 else ''} & {nm} & {r['comparisons']} & {r['constants']} & "
                     f"{100 * r['fidelity']:.1f} & {star(r['vs_network'])} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    write_table("explain.tex", L)


def tab_constrain():
    C = O["constrain"]
    N = {k: load("L2_rules", f"naive_net{k}")["tts"] for k in NETS}
    U = {k: load("L2", f"ppo_s{100 + k}")["tts"] for k in NETS}
    ov = SEC["overrides"]
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "Controller under the rules & Net 0 & Net 1 & Net 2 & Pooled vs.\\ naive & Departs from best legal (\\%) \\\\", "\\midrule",
         "Network without the rules (TTS) & " + " & ".join(f"{U[k].mean():.1f}" for k in NETS) + " & & \\\\",
         "Network + shield, naive (TTS) & " + " & ".join(f"{N[k].mean():.1f}" for k in NETS)
         + f" & {ci_txt(C['rules_cost'])}$^{{\\dagger}}$ & \\\\", "\\midrule"]
    for b, lab in ((0, "LLM supervisor, one shot"), (10, "LLM supervisor, 10 calls")):
        L.append(f"{lab} & " + " & ".join(star(C["runs"][f"net{k}_b{b}"]["vs_naive"], 2) for k in NETS)
                 + f" & {ci_txt(C['pooled'][str(b)], 2)} & "
                 + ", ".join(f"{ov[f'net{k}_b{b}']['departs_from_best_legal_pct']:.1f}" for k in NETS) + " \\\\")
    for key, fmt, lab in (("masked", "masked_net{k}", "Network, best legal phase"),
                          ("finetune_300k", "ft_net{k}_s70{k}_300k", "PPO fine-tuned under the rules, 300k"),
                          ("finetune_1M", "ft_net{k}_s70{k}_1000k", "PPO fine-tuned under the rules, 1M")):
        rs = [paired(load("L2_rules", fmt.format(k=k))["tts"], N[k]) for k in NETS]
        L.append(f"{lab} & " + " & ".join(star(r, 2) for r in rs) + f" & {ci_txt(C['baselines'][key], 2)} & \\\\")
    L.append(f"PPO trained under the rules, 3M (3 seeds) & \\multicolumn{{3}}{{r}}{{TTS "
             + ", ".join(f"{load('L2_rules', f'ppo_s{s}')['tts'].mean():.1f}" for s in range(300, 303))
             + f"}} & {ci_txt(C['baselines']['scratch_under_rules_vs_naive'], 2)} & \\\\")
    L.append("\\midrule")
    for lab, key in (("Actuated, tuned under the rules (TTS)", "actuated_under_rules"),
                     ("Fixed time, tuned under the rules (TTS)", "fixed_under_rules")):
        L.append(f"{lab} & \\multicolumn{{3}}{{r}}{{{C['baselines'][key]['all']:.1f}}} & & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("constrain.tex", L)


def tab_cost():
    s = SEC["cost"]["summary"]
    rows = [("init", "Initialise (one shot)"), ("transfer", "Transfer (one shot)"), ("explain", "Explain (one shot)"),
            ("patch/b0", "Patch, one shot"), ("patch/b5", "Patch, 5 calls"), ("patch/b10", "Patch, 10 calls"),
            ("patch_fault/b0", "Patch (fault), one shot"), ("patch_fault/b5", "Patch (fault), 5 calls"),
            ("patch_fault/b10", "Patch (fault), 10 calls"), ("constrain/b0", "Constrain, one shot"),
            ("constrain/b10", "Constrain, 10 calls"), ("patch_models/fable", "Patch, Fable 5.1, 10 calls"),
            ("patch_models/sonnet", "Patch, Sonnet 5.5, 10 calls"), ("patch_models/haiku", "Patch, Haiku 5.5, 10 calls")]
    L = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "Use (Opus 5.5 unless stated; 3 runs each) & Active min & Tool calls & Output tokens (k) & Simulator calls & Cost (USD) \\\\",
         "\\midrule"]
    for key, lab in rows:
        v = s[key]
        rng = lambda f, d=0: f"{v[f]['mean']:.{d}f} ({v[f]['min']:.{d}f}--{v[f]['max']:.{d}f})"  # noqa: E731
        L.append(f"{lab} & {rng('minutes')} & {v['tool_calls']['mean']:.0f} & {v['output_tokens']['mean'] / 1000:.0f} & "
                 f"{v['sim_calls']['mean']:.1f} & {v['usd']['mean']:.2f} \\\\")
    t = SEC["cost"]["total"]
    L.append("\\midrule")
    L.append(f"All {t['runs']} runs & {t['minutes']:.0f} & & {t['output_tokens'] / 1000:,.0f} & & {t['usd']:.2f} \\\\".replace(",", "{,}"))
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("agent_cost.tex", L)


def tab_convergence():
    conv = O["reference"]["convergence"]
    names = {"ppo_s": "PPO from scratch", "init_": "PPO from LLM network", "transfer_": "PPO from LLM transfer",
             "ft_": "PPO fine-tuning"}
    L = ["\\begin{tabular}{llrrrl}", "\\toprule", "Problem & Run & Steps & Final val.\\ TTS & Change, last 20\\,\\% & Converged \\\\",
         "\\midrule"]
    for key in sorted(conv, key=lambda k: (k.split("/")[0], k.split("/")[1])):
        c = conv[key]
        prob, run = key.split("/")
        L.append(f"{prob.replace('_', ' ')} & {run.replace('_', ' ')} & {c['steps'] / 1e6:g}M & {c['final_val']:.1f} & "
                 f"{pct(c['change_pct'])}\\,\\% & {'yes' if c['converged'] else 'no'} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("convergence.tex", L)


def tab_calibration():
    rows = O["calibration"]
    L = ["\\begin{tabular}{llrrr}", "\\toprule", "Use & Run & Low & Nominal & High \\\\", "\\midrule"]
    for r in rows:
        e = r["signed_error_pct"]
        L.append(f"{r['case'].replace('_', ' ')} & {r['run'].replace('_', ' ')} & " + " & ".join(pct(e[s]) for s in SC) + " \\\\")
    allv = np.array([[r["signed_error_pct"][s] for s in SC] for r in rows])
    L.append("\\midrule")
    L.append(f"All ({len(rows)} runs) & mean $|$error$|$ & " + " & ".join(f"{np.abs(allv[:, i]).mean():.1f}" for i in range(3)) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("calibration.tex", L)


def tab_cap():
    cap = SEC["cap"]["labels"]
    un = SEC["unserved"]
    keys = ["L2/ppo_s100", "L2/ppo_s102", "L2/ppo_s103", "L2/init_r2", "L2/patch_net2_b10", "L2/explain_net0_small",
            "L2/explain_net1_small", "L2/cart_net2_small", "L2W/ppo_s201", "L2W/pad_net0", "L2_fault/faulty_net1",
            "L2_fault/meanimp_net1", "L2_fault/patchf_net0_b0"]
    L = ["\\begin{tabular}{lrrrr}", "\\toprule",
         "Evaluation & Unserved vehicles (episodes) & TTS, 30-min cap & TTS, 60-min cap & Unserved, 60-min cap \\\\", "\\midrule"]
    for k in keys:
        if k not in cap:
            continue
        c = cap[k]
        L.append(f"{k.replace('_', ' ')} & {un[k]['total']} ({un[k]['episodes_with_unserved']}) & {c['tts_30min']:.1f} & "
                 f"{c['tts_60min']:.1f} & {c['unserved_60min']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write_table("cap.tex", L)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from wsp import plot_style as S
    S.apply()
    os.makedirs(FIGS, exist_ok=True)
    for f in (fig_demand, fig_reference, fig_curves, fig_patch, fig_fault, fig_explain, fig_constrain, fig_truncation):
        f(S, plt)
        print("figure", f.__name__)
    for t in (tab_reference, tab_warm, tab_patch, tab_fault, tab_fault_rep, tab_explain, tab_constrain, tab_cost, tab_convergence,
              tab_calibration, tab_cap):
        t()
        print("table", t.__name__)


if __name__ == "__main__":
    main()
