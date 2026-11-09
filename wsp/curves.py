# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Validation learning curve of a PPO run and the convergence criterion (EXPERIMENT_PLAN.md, common rules).

    python -m wsp.curves --run models/L2/ppo_s100 [--workers 6]

Evaluates every checkpoint (ckpt_<steps>_steps.zip and final.zip) on the 6 val episodes and writes
<run>/curve_val.json: {"steps": [...], "tts": [...], "unserved": [...], "converged": bool, "change_pct": x}.
Criterion: the mean validation TTS over the last 20 % of the run's steps is no more than 2 % below the mean over the
preceding 20 % (change_pct >= -2). Checkpoints already evaluated are reused.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from wsp.evaluate import SETS, run_many
from wsp.junctions import V3


def convergence(steps, tts, total):
    steps, tts = np.asarray(steps, float), np.asarray(tts, float)
    last = tts[(steps > 0.8 * total) & (steps <= total)]
    prev = tts[(steps > 0.6 * total) & (steps <= 0.8 * total)]
    if len(last) == 0 or len(prev) == 0:
        return None, None
    change = 100 * (last.mean() / prev.mean() - 1)
    return bool(change >= -2.0), float(change)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    cfg = json.load(open(os.path.join(run, "config.json")))
    problem, total = cfg["problem"], cfg["steps"]
    ck = {int(re.search(r"ckpt_(\d+)_steps", p).group(1)): p for p in glob.glob(os.path.join(run, "ckpt_*_steps.zip"))}
    if os.path.exists(os.path.join(run, "final.zip")):
        ck[total] = os.path.join(run, "final.zip")
    out = os.path.join(run, "curve_val.json")
    old = json.load(open(out)) if os.path.exists(out) else {"steps": [], "tts": [], "unserved": []}
    have = dict(zip(old["steps"], zip(old["tts"], old["unserved"])))
    todo = sorted(s for s in ck if s not in have)
    jobs, owners = [], []
    for s in todo:
        for sc, seed in SETS["val"]:
            jobs.append(dict(problem=problem, policy=f"sb3:{ck[s]}", scenario=sc, seed=seed,
                             workdir=os.path.join(V3, "runs", "curves", problem, os.path.basename(run), f"{s}_{sc}_{seed}")))
            owners.append(s)
    rows = run_many(jobs, a.workers) if jobs else []
    for s in todo:
        rs = [r for r, o in zip(rows, owners) if o == s]
        have[s] = (float(np.mean([r["tts_veh_h"] for r in rs])), int(sum(r["remaining"] for r in rs)))
    steps = sorted(have)
    tts = [have[s][0] for s in steps]
    conv, change = convergence(steps, tts, total)
    res = dict(problem=problem, run=os.path.basename(run), total_steps=total, steps=steps, tts=tts,
               unserved=[have[s][1] for s in steps], converged=conv, change_pct=change)
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(out + ".tmp", out)
    verdict = "too few checkpoints" if conv is None else (f"change over the last 20 % {change:+.2f} % -> "
                                                          + ("converged" if conv else "NOT converged"))
    print(f"{problem} {os.path.basename(run)}: final val TTS {tts[-1]:.2f}; {verdict}")


if __name__ == "__main__":
    main()
