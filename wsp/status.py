# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Progress and time left of the job queue.

    (from the repository root)  python -m wsp.status          # once
                watch -n 30 python -m wsp.status
"""
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.queue import budget, read_jobs, read_state, runner_alive  # noqa: E402

DEFAULT_RATE = 1500.0           # env steps / s per training run, until measured
DEFAULT_S = {"tune": 300.0, "eval": 60.0}


def train_progress(j, s):
    mon = os.path.join(ROOT, j["run_dir"], "monitor.csv")
    if not os.path.exists(mon):
        return 0, None
    rows = [l for l in open(mon).read().splitlines()[2:] if l.strip()]
    if not rows:
        return 0, None
    steps = sum(int(float(r.split(",")[1])) for r in rows)      # episode lengths (variable with --train-drain)
    t = float(rows[-1].split(",")[2])
    return steps, (steps / t if t > 0 else None)


def fmt(sec):
    if sec is None:
        return "?"
    sec = max(0, int(sec))
    h, m = divmod(sec // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


def main():
    jobs, st, now = read_jobs(), read_state(), time.time()
    cap = budget()
    by = {j["id"]: j for j in jobs}
    status = {j["id"]: st.get(j["id"], {}).get("status", "pending") for j in jobs}
    # measured rates / durations
    rates, durs = [], {"tune": [], "eval": []}
    for j in jobs:
        s = st.get(j["id"], {})
        if j["kind"] == "train" and s.get("status") in ("running", "done"):
            _, r = train_progress(j, s)
            if r:
                rates.append(r)
        elif s.get("status") == "done" and j["kind"] in durs:
            durs[j["kind"]].append(s["end"] - s["start"])
    rate = float(np.median(rates)) if rates else DEFAULT_RATE
    est = {k: (float(np.mean(v)) if v else DEFAULT_S[k]) for k, v in durs.items()}

    def dur(j):
        return j["steps"] / rate if j["kind"] == "train" else est[j["kind"]]

    running = [j for j in jobs if status[j["id"]] == "running"]
    used = sum(j["cores"] for j in running)
    counts = {k: sum(1 for v in status.values() if v == k) for k in ("done", "running", "pending", "failed", "blocked", "cancelled")}
    auto = ""
    try:
        if open(os.path.join(ROOT, "queue/budget")).read().startswith("auto"):
            auto = " (auto: " + open(os.path.join(ROOT, "queue/budget_now")).read().split(" ", 1)[1].strip() + ")"
    except (OSError, IndexError):
        pass
    print(f"queue  {time.strftime('%Y-%m-%d %H:%M:%S')}   cores {used:.1f}/{cap:.0f}{auto}   "
          + "  ".join(f"{k} {v}" for k, v in counts.items() if v))
    if not runner_alive():
        print("  !! queue runner is NOT running (bash scripts/run_queue.sh)")
    print()
    print(f"  {'running job':42s} {'progress':>22s} {'elapsed':>8s} {'left':>8s}")
    rem = {}
    for j in running:
        s = st[j["id"]]
        el = now - s["start"]
        if j["kind"] == "train":
            steps, r = train_progress(j, s)
            left = (j["steps"] - steps) / (r or rate)
            bar = int(20 * steps / j["steps"])
            prog = f"{steps / 1e6:4.2f}/{j['steps'] / 1e6:.1f}M {'#' * (bar // 2)}{'.' * (10 - bar // 2)}"
        else:
            left = max(est[j["kind"]] - el, 10)
            prog = j["kind"]
        rem[j["id"]] = left
        print(f"  {j['id']:42s} {prog:>22s} {fmt(el):>8s} {fmt(left):>8s}")
    # simulate the rest of the queue under the core budget
    pend = [j for j in jobs if status[j["id"]] == "pending"]
    t, active, done_sim = 0.0, [(rem[j["id"]], j) for j in running], {k for k, v in status.items() if v == "done"}
    finish_by_prefix = {}
    while pend or active:
        u = sum(j["cores"] for _, j in active)
        started = True
        while started:
            started = False
            for j in list(pend):
                if all(d in done_sim for d in j.get("after", [])) and (u + j["cores"] <= cap + 1e-9 or u == 0):
                    active.append((t + dur(j), j))
                    u += j["cores"]
                    pend.remove(j)
                    started = True
        if not active:
            break
        active.sort(key=lambda x: x[0])
        t, j = active.pop(0)
        done_sim.add(j["id"])
        stage = j.get("stage", j["id"].split("/")[0])
        finish_by_prefix[stage] = t
    if pend:
        print(f"\n  {len(pend)} pending jobs cannot start (dependencies failed/blocked)")
    print()
    for stage, tt in sorted(finish_by_prefix.items(), key=lambda kv: kv[1]):
        n_left = sum(1 for j in jobs if j.get("stage") == stage and status[j["id"]] in ("pending", "running"))
        n_all = sum(1 for j in jobs if j.get("stage") == stage)
        print(f"  stage {stage:28s} {n_all - n_left:3d}/{n_all:<3d} done   ETA {fmt(tt):>8s}  ({time.strftime('%H:%M', time.localtime(now + tt))})")
    total = max(finish_by_prefix.values()) if finish_by_prefix else 0
    print(f"\n  ALL QUEUED WORK: ~{fmt(total)} left  (finishes ~{time.strftime('%a %H:%M', time.localtime(now + total))}; "
          f"training rate {rate:.0f} steps/s per run)")
    failed = [k for k, v in status.items() if v == "failed"]
    if failed:
        print("  failed:", ", ".join(failed[:10]), "(see queue/logs/)")


if __name__ == "__main__":
    main()
