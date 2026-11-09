# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Paired statistics on the 30 held-out test episodes (docs/EXPERIMENT_PLAN.md, common rules).

    from wsp.stats import load, paired, pooled, welch, fmt_ci

load(problem, label)          -> {"tts": array(30), "unserved": array(30)} in SETS["test"] order, or None
paired(a, b)                  -> paired comparison of TTS vectors a vs b (a is the first policy): mean difference
                                 (veh-h) with its two-sided 95 % t-interval (29 df), the change in mean TTS relative
                                 to b's mean ("pct", the headline measure) with the same interval rescaled, and the
                                 number of episodes on which a is lower (better)
pooled([(a1, b1), ...])       -> the same after averaging the per-episode differences over targets / seeds
welch(xs, ys)                 -> independent families of runs: difference of per-run means with a Welch interval,
                                 also as % of mean(ys)
"""
import json
import os

import numpy as np
from scipy import stats

from wsp.evaluate import SETS, result_path

ORDER = SETS["test"]


def load(problem, label, set_name="test", drain_cap=1800):
    path = result_path(problem, label, set_name, drain_cap)
    if not os.path.exists(path):
        return None
    rows = {(r["scenario"], r["seed"]): r for r in json.load(open(path))["episodes"]}
    keys = SETS[set_name]
    return {"tts": np.array([rows[k]["tts_veh_h"] for k in keys]),
            "unserved": np.array([rows[k]["remaining"] for k in keys]),
            "delay": np.array([rows[k]["delay"] for k in keys])}


def _ci(d):
    n = len(d)
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h)


def _pack(d, base):
    m, lo, hi = _ci(d)
    k = 100.0 / base
    return {"diff": m, "lo": lo, "hi": hi, "pct": m * k, "pct_lo": lo * k, "pct_hi": hi * k,
            "wins": int((d < 0).sum()), "n": len(d), "sig": bool(lo > 0 or hi < 0)}


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = _pack(a - b, float(b.mean()))
    out.update(mean_a=float(a.mean()), mean_b=float(b.mean()))
    return out


def pooled(pairs):
    a = np.mean([np.asarray(x, float) for x, _ in pairs], axis=0)
    b = np.mean([np.asarray(y, float) for _, y in pairs], axis=0)
    d = np.mean([np.asarray(x, float) - np.asarray(y, float) for x, y in pairs], axis=0)
    out = _pack(d, float(b.mean()))
    out.update(mean_a=float(a.mean()), mean_b=float(b.mean()), k=len(pairs))
    return out


def welch(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    d = xs.mean() - ys.mean()
    vx, vy = xs.var(ddof=1) / len(xs), ys.var(ddof=1) / len(ys)
    se = np.sqrt(vx + vy)
    dof = (vx + vy) ** 2 / (vx ** 2 / (len(xs) - 1) + vy ** 2 / (len(ys) - 1)) if se > 0 else 1.0
    h = stats.t.ppf(0.975, dof) * se
    k = 100.0 / ys.mean()
    return {"diff": float(d), "lo": float(d - h), "hi": float(d + h), "pct": float(d * k), "pct_lo": float((d - h) * k),
            "pct_hi": float((d + h) * k), "dof": float(dof), "mean_a": float(xs.mean()), "mean_b": float(ys.mean()),
            "sd_a": float(xs.std(ddof=1)), "sd_b": float(ys.std(ddof=1)), "n_a": len(xs), "n_b": len(ys),
            "sig": bool(d - h > 0 or d + h < 0)}


def fmt_ci(r, pct=True, digits=1):
    if pct:
        return f"{r['pct']:+.{digits}f}\\,\\% [{r['pct_lo']:+.{digits}f}, {r['pct_hi']:+.{digits}f}]"
    return f"{r['diff']:+.{digits}f} [{r['lo']:+.{digits}f}, {r['hi']:+.{digits}f}]"
