# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Secondary results (EXPERIMENT_PLAN.md, P "Secondary", C, appendix checks) -> results/secondary.json + SECONDARY.md.

    (from the repository root)  python -m wsp.secondary [--evaluate]

    truncation   closed-loop runs (Patch B = 5 / 10, failed-detector Patch B = 5 / 10, Constrain B = 10): after each
                 call k, the candidate with the best dev TTS so far, scored on test (only candidates whose file still
                 has the logged hash). Labels cand_<case>_<id>_<sha>.
    edits        submitted Patch networks against their target: parameters changed per layer, units touched, relative
                 size of the change.
    cost         agent runs (active minutes, tool calls, output tokens, USD, simulator calls) and the simulation used by
                 the non-LLM baselines (episodes, steps, wall-clock).
    overrides    Constrain supervisors on test (recorded rerun of the scored label): how often the supervisor departs
                 from the network's best legal phase, and how often the network's own choice is illegal.
    unserved     vehicles not served at the 30-min drain cap, every test evaluation.
    cap          the comparisons that involve unserved vehicles, re-scored with a 60-min cap.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

from wsp.evaluate import V3, evaluate, result_path, SETS
from wsp.policies import forward, load_net
from wsp.stats import load, paired, pooled
from wsp.paths import resolve

PACKS = os.path.realpath(os.path.join(V3, "packs"))
NETS = (0, 1, 2)
LAYERS = ("W1", "b1", "W2", "b2", "W3", "b3")


def r(x, d=4):
    return None if x is None else float(np.round(x, d))


def target_zip(k):
    return os.path.join(V3, "models", "L2", f"ppo_s{100 + k}", "final.zip")


def sha(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def calls_of(case, pid):
    p = os.path.join(PACKS, case, pid, "eval_log.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


# ---------------------------------------------------------------- truncation
CASES = {"patch": ("L2", "npz:{f}", "patch_{pid}"),
         "patch_fault": ("L2_fault", "npz:{f}", "patchf_{pid}"),
         "constrain": ("L2_rules", "super:{f}|{net}", "constrain_{pid}")}


def truncation(do_eval):
    out = {}
    runs = ([("patch", f"net{k}_b{b}") for b in (5, 10) for k in NETS]
            + [("patch_fault", f"net{k}_b{b}") for b in (5, 10) for k in NETS]
            + [("constrain", f"net{k}_b10") for k in NETS])
    for case, pid in runs:
        prob, spec, sub_lab = CASES[case]
        k = int(pid[3])
        ref_lab = {"patch": f"ppo_s{100 + k}", "patch_fault": f"faulty_net{k}", "constrain": f"naive_net{k}"}[case]
        ref_prob = "L2" if case == "patch" else prob
        ref = load(ref_prob, ref_lab)["tts"]
        healthy = load("L2", f"ppo_s{100 + k}")["tts"] if case == "patch_fault" else None
        sub = load(prob, sub_lab.format(pid=pid))["tts"]
        calls = calls_of(case, pid)
        steps = []
        for n in range(1, len(calls) + 1):
            i = min(range(n), key=lambda j: calls[j]["tts_mean"])
            c = calls[i]
            intact = (sha(resolve(c["file"])) or "")[:len(c["sha1"])] == c["sha1"]
            row = {"after_call": n, "best_call": i + 1, "dev_tts": r(c["tts_mean"]), "intact": intact}
            if intact:
                lab = f"cand_{case}_{pid}_{c['sha1']}"
                pol = spec.format(f=c["file"], net=target_zip(k))
                if do_eval:
                    evaluate(prob, pol, "test", lab, workers=10)
                x = load(prob, lab)
                if x is not None:
                    row["test"] = r(x["tts"].mean())
                    row["vs_ref"] = {kk: r(v) for kk, v in paired(x["tts"], ref).items() if kk in ("pct", "pct_lo", "pct_hi")}
                    if healthy is not None:
                        row["recovered_pct"] = r(100 * (ref.mean() - x["tts"].mean()) / (ref.mean() - healthy.mean()), 2)
            steps.append(row)
        # the submitted file may be a re-saved copy of an evaluated candidate: match on the 30 test episodes
        sub_sha = sha(os.path.join(PACKS, case, pid, "work", "supervisor.py" if case == "constrain" else "weights.npz"))
        same = []
        for j, c in enumerate(calls):
            x = load(prob, f"cand_{case}_{pid}_{c['sha1']}")
            if (sub_sha and sub_sha.startswith(c["sha1"])) or (
                    x is not None and np.allclose(x["tts"], sub, rtol=0, atol=1e-9)):
                same.append(j + 1)
        best = min(range(len(calls)), key=lambda j: calls[j]["tts_mean"]) + 1 if calls else None
        out[f"{case}/{pid}"] = {
            "calls_used": len(calls), "budget": int(pid.split("_b")[1]), "steps": steps,
            "submitted_test": r(sub.mean()), "submitted_vs_ref": {kk: r(v) for kk, v in paired(sub, ref).items()
                                                                   if kk in ("pct", "pct_lo", "pct_hi")},
            "best_dev_call": best, "submitted_matches_calls": same,
            "submitted_is_best_dev": bool(best in same) if steps and steps[-1]["intact"] else None,
            "ref": ref_lab}
    return out


# ---------------------------------------------------------------- edits
def edits():
    out = {}
    packs = ([("patch", f"net{k}_b{b}") for k in NETS for b in (0, 5, 10)]
             + [("patch_fault", f"net{k}_b{b}") for k in NETS for b in (0, 5, 10)]
             + [("patch_models", f"net{k}_{m}") for k in NETS for m in ("sonnet", "haiku", "fable")])
    for case, pid in packs:
        p = os.path.join(PACKS, case, pid, "work", "weights.npz")
        if not os.path.exists(p):
            continue
        k = int(pid[3])
        w0, w = load_net(target_zip(k)), dict(np.load(p))
        row = {}
        for L in LAYERS:
            d = np.asarray(w[L], float) - w0[L]
            ch = np.abs(d) > 0
            row[L] = {"changed": int(ch.sum()), "of": int(ch.size),
                      "rel_change": r(np.linalg.norm(d) / max(np.linalg.norm(w0[L]), 1e-12)),
                      "max_abs": r(np.abs(d).max())}
            if L.startswith("W"):
                row[L]["rows_touched"] = int(ch.any(axis=1).sum())
                row[L]["cols_touched"] = int(ch.any(axis=0).sum())
        out[f"{case}/{pid}"] = {"total_changed": int(sum(v["changed"] for v in row.values())), "layers": row}
    return out


# ---------------------------------------------------------------- cost
def cost():
    runs = json.load(open(os.path.join(V3, "runs", "agent_runs.json")))
    agents = {}
    for a in runs:
        use = a["use"]
        arm = a["run"].split("_", 1)[1] if "_" in a["run"] else "-"
        key = f"{use}/{arm}" if use in ("patch", "patch_fault", "constrain", "patch_models") else use
        agents.setdefault(key, []).append({"run": a["run"], "minutes": a["active_minutes"],
                                           "tool_calls": int(sum(a["tool_calls"].values())),
                                           "output_tokens": a["output_tokens"], "usd": r(a["cost_usd"], 2),
                                           "sim_calls": a["simulator_calls_logged"], "turns": a["turns"],
                                           "models": a["models"], "flags": a["flags"]})
    summ = {k: {f: {"mean": r(np.mean([x[f] for x in v]), 2), "min": r(min(x[f] for x in v), 2),
                    "max": r(max(x[f] for x in v), 2)} for f in ("minutes", "tool_calls", "output_tokens", "usd", "sim_calls")}
            | {"n": len(v)} for k, v in agents.items()}
    total = {"runs": len(runs), "usd": r(sum(a["cost_usd"] for a in runs), 2),
             "minutes": r(sum(a["active_minutes"] for a in runs), 1),
             "output_tokens": int(sum(a["output_tokens"] for a in runs))}

    def mon(run_dir, upto=None):
        p = os.path.join(run_dir, "monitor.csv")
        if not os.path.exists(p):
            return None
        rows = [l.split(",") for l in open(p).read().splitlines()[2:] if l.strip()]
        L = np.array([int(float(x[1])) for x in rows])
        T = np.array([float(x[2]) for x in rows])
        cum = np.cumsum(L)
        m = cum <= (upto or cum[-1])
        return {"episodes": int(m.sum()), "steps": int(cum[m][-1]), "wall_min": r(T[m][-1] / 60, 1),
                "sim_hours": r(cum[m][-1] * 5 / 3600, 1)}

    base = {}
    for name, pat in (("ppo_L2", "models/L2/ppo_s10?"), ("ppo_L2W", "models/L2W/ppo_s20?"),
                      ("ppo_rules", "models/L2_rules/ppo_s30?"), ("ft_patch", "models/L2/ft_net?_s60?"),
                      ("ft_fault", "models/L2_fault/ft_net?_s80?"), ("ft_rules", "models/L2_rules/ft_net?_s70?")):
        dirs = sorted(glob.glob(os.path.join(V3, pat)))
        rows = {os.path.basename(d): mon(d) for d in dirs}
        if name.startswith("ft"):
            rows300 = {os.path.basename(d) + "@300k": mon(d, 300_000) for d in dirs}
            rows.update(rows300)
        base[name] = rows
    dev_eps = len(SETS["dev"])
    base["es"] = {"episodes_per_run": 10 * dev_eps,
                  "episodes_step_size_selection": 3 * 2 * 10 * dev_eps,
                  "note": "10 dev evaluations per ES run; step-size selection runs 3 sigma x 2 seeds per target"}
    base["agent_episodes_per_call"] = dev_eps
    return {"agents": agents, "summary": summ, "total": total, "baselines": base}


# ---------------------------------------------------------------- overrides (Constrain)
def overrides(do_eval):
    out = {}
    for k in NETS:
        w = load_net(target_zip(k))
        for b in (0, 10):
            pid = f"net{k}_b{b}"
            sup = os.path.join(PACKS, "constrain", pid, "work", "supervisor.py")
            lab = f"constrain_{pid}"
            base = os.path.join(V3, "runs", "eval", "L2_rules", lab, "test")
            if do_eval and not glob.glob(os.path.join(base, "*", "record.npz")):
                d = evaluate("L2_rules", f"super:{sup}|{target_zip(k)}", "test", lab, workers=10, record=True)
                cached = load("L2_rules", lab)["tts"]
                rerun = {(e["scenario"], e["seed"]): e["tts_veh_h"] for e in d["episodes"]}
                out.setdefault("_determinism", {})[lab] = r(max(abs(rerun[key] - cached[i])
                                                                for i, key in enumerate(SETS["test"])), 6)
            n = free = dep = dep_greedy = illegal_net = illegal_sup = 0
            new_min = overdue = early = 0
            for f in glob.glob(os.path.join(base, "*", "record.npz")):
                z = np.load(f)
                X, A, S = z["obs"].astype(float), z["act"][:, 0], z["rule_state"]
                legal = S[:, 4:8] > 0.5
                lg = forward(w, X)
                greedy = lg.argmax(1)
                best_legal = np.where(legal, lg, -np.inf).argmax(1)
                choice = legal.sum(1) > 1
                n += len(A)
                free += int(choice.sum())
                dep += int((A != best_legal)[choice].sum())
                dep_greedy += int((A != greedy).sum())
                illegal_net += int((~legal[np.arange(len(A)), greedy]).sum())
                illegal_sup += int((~legal[np.arange(len(A)), A]).sum())
                il = ~legal[np.arange(len(A)), greedy]
                tg, g = S[:, 16], S[:, 17].astype(int)
                nm = il & np.isin(g, [0, 2]) & (tg >= 10) & (tg < 15)       # the new 15 s minimum binds
                early += int((il & (tg < 10)).sum())                         # ignored under the old rules too
                new_min += int(nm.sum())
                overdue += int((il & (tg >= 10) & ~nm & (S[:, 8:12] > 0.5).any(1)).sum())
            if n:
                out[pid] = {"steps": n, "steps_with_choice": free,
                            "departs_from_best_legal_pct": r(100 * dep / max(free, 1), 2),
                            "departs_from_greedy_pct": r(100 * dep_greedy / n, 2),
                            "network_choice_illegal_pct": r(100 * illegal_net / n, 2),
                            "supervisor_choice_illegal_pct": r(100 * illegal_sup / n, 2),
                            "network_breaks_new_min_green_pct": r(100 * new_min / n, 2),
                            "network_ignores_overdue_pct": r(100 * overdue / n, 2),
                            "network_switch_before_10s_pct": r(100 * early / n, 2)}
    return out


# ---------------------------------------------------------------- unserved / cap
def unserved():
    out = {}
    for p in sorted(glob.glob(os.path.join(V3, "results", "eval", "*", "*", "test.json"))):
        d = json.load(open(p))
        u = [e["remaining"] for e in d["episodes"]]
        out[f"{d['problem']}/{d['label']}"] = {"total": int(sum(u)), "episodes_with_unserved": int(sum(1 for x in u if x > 0)),
                                             "max": int(max(u))}
    return out


def cap_check(un, do_eval, cap=3600):
    """Re-score every policy with unserved vehicles (and its reference) with a 60-min cap; report whether the headline
    comparison changes sign or significance."""
    involved = sorted(k for k, v in un.items() if v["total"] > 0 and not k.split("/")[1].startswith("cand_"))
    out = {"cap_s": cap, "labels": {}}
    for key in involved:
        prob, lab = key.split("/", 1)
        d = json.load(open(result_path(prob, lab, "test")))
        if do_eval:
            evaluate(prob, d["policy"], "test", lab, workers=10, drain_cap=cap)
        x = load(prob, lab, drain_cap=cap)
        if x is None:
            continue
        x0 = load(prob, lab)
        out["labels"][key] = {"tts_30min": r(x0["tts"].mean()), "tts_60min": r(x["tts"].mean()),
                              "unserved_30min": int(x0["unserved"].sum()), "unserved_60min": int(x["unserved"].sum())}
    return out


def report(o):
    L = ["# v3 secondary results\n", "## Truncation (best dev candidate after k calls, on test)\n"]
    for key, t in o["truncation"].items():
        s = " | ".join(f"k={x['after_call']}: c{x['best_call']} "
                       + (f"{x['vs_ref']['pct']:+.1f} %" if "vs_ref" in x else "n/a")
                       + (f" ({x['recovered_pct']:.0f} %)" if "recovered_pct" in x else "") for x in t["steps"])
        L.append(f"- {key} (vs {t['ref']}; {t['calls_used']}/{t['budget']} calls; submitted {t['submitted_vs_ref']['pct']:+.1f} %, "
                 f"best dev call {t['best_dev_call']}, submitted = call {t['submitted_matches_calls'] or 'none evaluated'}): {s}")
    L.append("\n## Edits (parameters changed per layer)\n")
    for key, e in o["edits"].items():
        L.append(f"- {key}: {e['total_changed']} | " + ", ".join(
            f"{l} {v['changed']}" + (f" ({v['rows_touched']} rows)" if "rows_touched" in v else "")
            for l, v in e["layers"].items() if v["changed"]))
    L.append("\n## Agent cost (mean [min, max] per run)\n")
    for key, s in o["cost"]["summary"].items():
        L.append(f"- {key} (n={s['n']}): {s['minutes']['mean']:.0f} min [{s['minutes']['min']:.0f}, {s['minutes']['max']:.0f}], "
                 f"{s['output_tokens']['mean'] / 1000:.0f}k output tokens, ${s['usd']['mean']:.2f}, "
                 f"{s['tool_calls']['mean']:.0f} tool calls, {s['sim_calls']['mean']:.1f} simulator calls")
    t = o["cost"]["total"]
    L.append(f"- total: {t['runs']} runs, ${t['usd']:.2f}, {t['minutes']:.0f} active min")
    L.append("\n## Constrain supervisors on test\n")
    for key, v in o["overrides"].items():
        if not key.startswith("_"):
            L.append(f"- {key}: departs from best legal phase {v['departs_from_best_legal_pct']:.1f} % of steps with a choice; "
                     f"network's own choice illegal {v['network_choice_illegal_pct']:.1f} %; supervisor illegal "
                     f"{v['supervisor_choice_illegal_pct']:.1f} %")
    if "_determinism" in o["overrides"]:
        L.append(f"- rerun determinism (max |dTTS|): {o['overrides']['_determinism']}")
    L.append("\n## Unserved vehicles at the 30-min cap (test; nonzero only)\n")
    for key, v in o["unserved"].items():
        if v["total"]:
            L.append(f"- {key}: {v['total']} vehicles in {v['episodes_with_unserved']}/30 episodes (max {v['max']})")
    L.append(f"\n## Cap check ({o['cap']['cap_s']} s)\n")
    for key, v in o["cap"]["labels"].items():
        L.append(f"- {key}: TTS {v['tts_30min']:.2f} -> {v['tts_60min']:.2f}; unserved {v['unserved_30min']} -> {v['unserved_60min']}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    o = {"truncation": truncation(a.evaluate), "edits": edits(), "cost": cost(), "overrides": overrides(a.evaluate)}
    o["unserved"] = unserved()
    o["cap"] = cap_check(o["unserved"], a.evaluate)
    json.dump(o, open(os.path.join(V3, "results", "secondary.json"), "w"), indent=1, default=float)
    txt = report(o)
    open(os.path.join(V3, "results", "SECONDARY.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
