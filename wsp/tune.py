# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Tune the rule-based reference controllers of a problem on its 9 dev episodes, then score them on val and test.

    python -m wsp.tune --problem L2 --workers 16

Fixed time (cycle 0 -> 1 -> 2 -> 3): greens (NS through, NS left, EW through, EW left) in
    {10, 15, 20, 25, 30} x {10, 15, 20} x {10, 15, 20, 25, 30, 40} x {10, 15, 20} s           (270 plans)
Actuated: ext in {1, 2, 3, 4, 5, 6, 8, 10, 12, 16} x max_green in {10, 15, 20, 25, 30, 40, 50, 60, 90} s
    x detector layout in {stop, up}                                                         (180 settings)
The setting with the lowest mean dev TTS is kept (ties: first in grid order). Under the rules ("L2_rules") the
same grids are searched with the rules enforced. Writes results/tuned/<problem>.json (refuses to overwrite) and
the val / test evaluations as labels "fixed" and "actuated".
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

from wsp.evaluate import SETS, evaluate, run_many
from wsp.junctions import V3

FIXED = dict(ns_t=[10, 15, 20, 25, 30], ns_l=[10, 15, 20], ew_t=[10, 15, 20, 25, 30, 40], ew_l=[10, 15, 20])
EXT = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
MAXG = [10, 15, 20, 25, 30, 40, 50, 60, 90]
LAYOUT = ["stop", "up"]


def grid_search(problem, specs, workers, tag):
    jobs = []
    for i, spec in enumerate(specs):
        for sc, s in SETS["dev"]:
            jobs.append(dict(problem=problem, policy=spec, scenario=sc, seed=s,
                             workdir=os.path.join(V3, "runs", "tune", problem, tag, f"{i}_{sc}_{s}")))
    rows = run_many(jobs, workers)
    score = {}
    for r in rows:
        score.setdefault(r["policy"], []).append(r["tts_veh_h"])
    ranked = sorted(specs, key=lambda sp: np.mean(score[sp]))          # stable: ties keep grid order
    return ranked[0], [(sp, float(np.mean(score[sp]))) for sp in ranked[:10]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    out = os.path.join(V3, "results", "tuned", f"{a.problem}.json")
    if os.path.exists(out):
        sys.exit(f"{out} exists; not overwriting")
    f = FIXED
    fixed = [f"fixed:{w},{x},{y},{z}" for w, x, y, z in itertools.product(f["ns_t"], f["ns_l"], f["ew_t"], f["ew_l"])]
    act = [f"actuated:{e},{m},{l}" for l, e, m in itertools.product(LAYOUT, EXT, MAXG)]
    res = {}
    for kind, specs in (("fixed", fixed), ("actuated", act)):
        best, top = grid_search(a.problem, specs, a.workers, kind)
        res[kind] = dict(spec=best, dev_tts=top[0][1], top=top, n_settings=len(specs))
        print(a.problem, kind, best, [(s, round(v, 2)) for s, v in top[:3]], flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x") as fh:
        json.dump(res, fh, indent=1)
    for kind, d in res.items():
        for st in ("val", "test"):
            r = evaluate(a.problem, d["spec"], st, kind, workers=a.workers)
            print(a.problem, kind, st, round(r["summary"]["all"]["tts_veh_h"], 2), flush=True)


if __name__ == "__main__":
    main()
