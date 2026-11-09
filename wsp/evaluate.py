# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Evaluate a controller on a fixed set of episodes, one SUMO process per episode.

Episode sets (docs/EXPERIMENT_PLAN.md):
    dev    low / nominal / high x seeds 1101-1103 (9)   baseline tuning, the agents' eval.py, wrapper / ES evaluations
    val    low / nominal / high x seeds 1301-1302 (6)   learning curves, convergence, ES step size, Explain held-out
    rec    training family x seeds 1201-1206 (6)        recorded states for task packs, DAgger rollouts
    cost   training family x seeds 1501-1560 (60)       reference controllers' training-cost distribution
    test   low / nominal / high x seeds 19001-19010 (30) final scoring only

Every episode runs 1 h of arrivals and drains (cap +30 min). KPIs: tts_veh_h (vehicles in the network plus vehicles
waiting to enter, summed every second); delay (mean time loss + departure delay per finished trip, s, from SUMO's
statistic output); remaining (vehicles not served at the cap).

    python -m wsp.evaluate --problem L2 --policy actuated --set test --label actuated
Results are cached in results/eval/<problem>/<label>/<set>.json; a cached file is reused only if its policy
string matches, and never overwritten.
"""
import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from wsp.junctions import V3
from wsp.paths import same_spec

SC = ("low", "nominal", "high")
SETS = {"dev": [(sc, s) for sc in SC for s in (1101, 1102, 1103)],
        "val": [(sc, s) for sc in SC for s in (1301, 1302)],
        "rec": [("train", s) for s in range(1201, 1207)],
        "cost": [("train", s) for s in range(1501, 1561)],
        "test": [(sc, s) for sc in SC for s in range(19001, 19011)]}
KEYS = ("tts_veh_h", "delay", "delay_veh_h", "halting_veh_h", "remaining", "arrived")


def _parse_stats(path):
    r = ET.parse(path).getroot()
    v, trip = r.find("vehicles"), r.find("vehicleTripStatistics")
    return {"loaded": int(v.get("loaded")), "inserted": int(v.get("inserted")), "arrived": int(float(trip.get("count"))),
            "delay": float(trip.get("timeLoss")) + float(trip.get("departDelay")),
            "travel": float(trip.get("duration")), "time_loss": float(trip.get("timeLoss")),
            "depart_delay": float(trip.get("departDelay"))}


def safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def run_episode(job):
    """job = dict(problem, policy, scenario, seed, workdir, record, drain_cap) -> KPI dict"""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from wsp.env import SignalEnv
    from wsp.policies import make_policy
    wd = job["workdir"]
    os.makedirs(wd, exist_ok=True)
    stats = os.path.join(wd, "statistics.xml")
    env = SignalEnv(job["problem"], scenario=job["scenario"], seed=job["seed"], drain=True,
                    drain_cap=job.get("drain_cap", 1800), statistic_output=stats, workdir=wd, obs=job.get("obs", "stdup"),
                    fault=job.get("fault"))
    env.fixed_sumo_seed = job["seed"]
    pol = make_policy(job["policy"], job["problem"])
    obs, _ = env.reset(seed=job["seed"])
    rec_o, rec_a, rec_s = [], [], []
    done, info = False, {}
    while not done:
        a = pol.act(obs, env)
        if job.get("record"):
            rec_o.append(obs)
            rec_a.append(np.asarray(a))
            if env.rules:
                st = env.shield_state(0)
                rec_s.append(np.r_[st["since_served"], st["legal"], st["overdue"], st["phase_halting"],
                                   st["t_green"], st["green"]])
        obs, r, done, _, info = env.step(a)
    env.close()
    out = dict(problem=job["problem"], policy=job["policy"], scenario=job["scenario"], seed=job["seed"],
               **info, **_parse_stats(stats))
    if job.get("record"):
        extra = {"rule_state": np.array(rec_s, np.float32)} if rec_s else {}
        np.savez_compressed(os.path.join(wd, "record.npz"), obs=np.array(rec_o, np.float32),
                            act=np.array(rec_a, np.int8), **extra)
    return out


def run_many(jobs, workers):
    if workers <= 1 or len(jobs) == 1:
        return [run_episode(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return list(ex.map(run_episode, jobs))


def jobs_for(problem, policy, set_name, label, record=False, drain_cap=1800, obs="stdup"):
    base = os.path.join(V3, "runs", "eval", problem, safe(label), set_name + ("_cap%d" % drain_cap if drain_cap != 1800 else ""))
    return [dict(problem=problem, policy=policy, scenario=sc, seed=s, record=record, drain_cap=drain_cap, obs=obs,
                 workdir=os.path.join(base, f"{sc}_{s}")) for sc, s in SETS[set_name]]


def summary(rows):
    out = {}
    for sc in sorted({r["scenario"] for r in rows}, key=lambda x: (SC + ("train",)).index(x)):
        rs = [r for r in rows if r["scenario"] == sc]
        out[sc] = {k: float(np.mean([r[k] for r in rs])) for k in KEYS}
        out[sc]["n"] = len(rs)
    out["all"] = {k: float(np.mean([r[k] for r in rows])) for k in KEYS}
    out["all"]["unserved"] = int(sum(r["remaining"] for r in rows))
    return out


def result_path(problem, label, set_name, drain_cap=1800):
    suffix = "" if drain_cap == 1800 else f"_cap{drain_cap}"
    return os.path.join(V3, "results", "eval", problem, safe(label), f"{set_name}{suffix}.json")


def evaluate(problem, policy, set_name, label, workers=8, record=False, drain_cap=1800, obs="stdup"):
    """Cached evaluation -> dict(problem, policy, set, label, summary, episodes)."""
    path = result_path(problem, label, set_name, drain_cap)
    if os.path.exists(path) and not record:
        d = json.load(open(path))
        if not same_spec(d["policy"], policy):
            raise RuntimeError(f"{path} holds policy {d['policy']!r}, not {policy!r}")
        return d
    rows = run_many(jobs_for(problem, policy, set_name, label, record, drain_cap, obs), workers)
    d = dict(problem=problem, policy=policy, set=set_name, label=label, drain_cap=drain_cap,
             summary=summary(rows), episodes=rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path + ".tmp", "w") as f:
            json.dump(d, f, indent=1)
        os.replace(path + ".tmp", path)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--set", default="val", choices=list(SETS))
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--drain-cap", type=int, default=1800)
    a = ap.parse_args()
    d = evaluate(a.problem, a.policy, a.set, a.label, a.workers, a.record, a.drain_cap)
    for sc, s in d["summary"].items():
        print(f"{a.problem:9s} {a.label:28s} {sc:8s} TTS {s['tts_veh_h']:7.2f}  delay {s['delay']:6.1f} s  "
              f"remaining {s['remaining']:.2f}")


if __name__ == "__main__":
    main()
