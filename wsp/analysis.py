# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""All pre-registered comparisons of docs/EXPERIMENT_PLAN.md -> results/paper_numbers.json and results/RESULTS.md.

    (from the repository root)  python -m wsp.analysis [--evaluate]      # --evaluate first scores the tree baselines on test

Conventions: paired comparisons over the 30 test episodes (wsp.stats.paired / pooled), Welch intervals over per-run
means for independent families of runs, every run reported, final checkpoints only.
"""
import argparse
import glob
import json
import os

import numpy as np

from wsp.evaluate import V3, evaluate
from wsp.stats import load, paired, pooled, welch

PACKS = os.path.realpath(os.path.join(V3, "packs"))
SC = {"low": slice(0, 10), "nominal": slice(10, 20), "high": slice(20, 30)}
NETS = (0, 1, 2)


def r2(x):
    return float(np.round(x, 4))


def ci(r):
    return {k: r2(v) if isinstance(v, float) else v for k, v in r.items()}


def per_sc(x):
    return {s: r2(x[sl].mean()) for s, sl in SC.items()} | {"all": r2(x.mean())}


def target(k, p="L2"):
    return f"ppo_s{100 + k}"


def scored(case, pid):
    p = os.path.join(PACKS, case, pid, "scored.json")
    return json.load(open(p)) if os.path.exists(p) else None


def monitor_delay(run_dir):
    """Per training episode: (length in steps, total delay veh-h) from monitor.csv (raw returns)."""
    rows = [l.split(",") for l in open(os.path.join(run_dir, "monitor.csv")).read().splitlines()[2:] if l.strip()]
    R = np.array([float(r[0]) for r in rows])
    L = np.array([int(float(r[1])) for r in rows])
    return L, -50.0 * R / 3600.0


def cost_stats(run_dir, thr, first_eps=50, first_steps=1_000_000):
    L, D = monitor_delay(run_dir)
    cum = np.cumsum(L)
    m = cum <= first_steps
    return {"first50_mean_delay": r2(D[:first_eps].mean()), "first1M_total_delay": r2(D[m].sum()),
            "first1M_episodes": int(m.sum()), "unacceptable_first200": int((D[:200] > thr).sum()),
            "unacceptable_first1M": int((D[m] > thr).sum()), "all_episodes": int(len(D)),
            "unacceptable_all": int((D > thr).sum())}


def reference(out):
    res = {}
    for p, seeds in (("L2", range(100, 105)), ("L2W", [200, 201, 202, "203_r1", 204]), ("L2_rules", range(300, 303))):
        fx, ac = load(p, "fixed"), load(p, "actuated")
        runs = {s: load(p, f"ppo_s{s}") for s in seeds}
        runs = {s: v for s, v in runs.items() if v is not None}
        d = {"fixed": per_sc(fx["tts"]), "actuated": per_sc(ac["tts"]),
             "ppo": {str(s): per_sc(v["tts"]) | {"unserved": int(v["unserved"].sum())} for s, v in runs.items()},
             "ppo_mean": per_sc(np.mean([v["tts"] for v in runs.values()], axis=0)),
             "ppo_sd_runs": r2(np.std([v["tts"].mean() for v in runs.values()], ddof=1)),
             "ppo_vs_actuated": ci(pooled([(v["tts"], ac["tts"]) for v in runs.values()])),
             "ppo_vs_fixed": ci(pooled([(v["tts"], fx["tts"]) for v in runs.values()])),
             "ppo_vs_actuated_per_seed": {str(s): ci(paired(v["tts"], ac["tts"])) for s, v in runs.items()},
             "actuated_vs_fixed": ci(paired(ac["tts"], fx["tts"]))}
        d["tuned"] = json.load(open(os.path.join(V3, "results", "tuned", f"{p}.json")))
        d["tuned"] = {k: v["spec"] for k, v in d["tuned"].items()}
        res[p] = d
    conv = {}
    for f in sorted(glob.glob(os.path.join(V3, "models", "*", "*", "curve_val.json"))):
        c = json.load(open(f))
        conv[f"{c['problem']}/{c['run']}"] = {"steps": c["total_steps"], "change_pct": c["change_pct"],
                                              "converged": c["converged"], "final_val": r2(c["tts"][-1])}
    res["convergence"] = conv
    out["reference"] = res


def warm_family(problem, prefix_fmt, nets, seeds, scratch_seeds, thr, label_zero):
    fx, ac = load(problem, "fixed"), load(problem, "actuated")
    d = {"zero_shot": {}, "warm": {}, "scratch": {}}
    for k in nets:
        z = load(problem, label_zero.format(k=k))
        if z is None:
            continue
        d["zero_shot"][str(k)] = {"tts": per_sc(z["tts"]), "unserved": int(z["unserved"].sum()),
                                  "vs_fixed": ci(paired(z["tts"], fx["tts"])), "vs_actuated": ci(paired(z["tts"], ac["tts"]))}
        for s in seeds:
            name = prefix_fmt.format(k=k, s=s)
            run = os.path.join(V3, "models", problem, name)
            t = load(problem, name)
            if not os.path.exists(os.path.join(run, "monitor.csv")):
                continue
            cv = json.load(open(os.path.join(run, "curve_val.json"))) if os.path.exists(os.path.join(run, "curve_val.json")) else {}
            d["warm"][name] = cost_stats(run, thr) | {"test": per_sc(t["tts"]) if t else None,
                                                       "converged": cv.get("converged"), "change_pct": cv.get("change_pct")}
    for s in scratch_seeds:
        run = os.path.join(V3, "models", problem, f"ppo_s{s}")
        t = load(problem, f"ppo_s{s}")
        d["scratch"][str(s)] = cost_stats(run, thr) | {"test": per_sc(t["tts"]) if t else None}
    w = [v for v in d["warm"].values()]
    sc = list(d["scratch"].values())
    if w and sc:
        for key in ("first50_mean_delay", "first1M_total_delay"):
            d[f"warm_vs_scratch_{key}"] = ci(welch([x[key] for x in w], [x[key] for x in sc]))
        if all(x["test"] for x in w):
            d["warm_vs_scratch_final_tts"] = ci(welch([x["test"]["all"] for x in w], [x["test"]["all"] for x in sc]))
    d["threshold_unacceptable"] = r2(thr)
    return d


def cost_threshold(problem):
    d = json.load(open(os.path.join(V3, "results", "eval", problem, "fixed", "cost.json")))
    return float(np.percentile([e["delay_veh_h"] for e in d["episodes"]], 95))


def initialise(out):
    d = warm_family("L2", "init_r{k}_s{s}", (1, 2, 3), (400, 401), range(100, 105), cost_threshold("L2"), "init_r{k}")
    for k in (1, 2, 3):
        s = scored("init", f"r{k}")
        if s:
            d["zero_shot"].setdefault(str(k), {})["trainability"] = s["trainability"]
            d["zero_shot"][str(k)]["predictions"] = s["predictions"]
    out["initialise"] = d


def transfer(out):
    seeds = [200, 201, 202, "203_r1", 204]
    d = warm_family("L2W", "transfer_net{k}_s{s}", NETS, (500, 501), seeds, cost_threshold("L2W"), "transfer_net{k}")
    scratch_means = [load("L2W", f"ppo_s{s}")["tts"].mean() for s in seeds]
    for k in NETS:
        z = load("L2W", f"transfer_net{k}")
        pad = load("L2W", f"pad_net{k}")
        if z is None:
            continue
        d["zero_shot"][str(k)]["vs_pad"] = ci(paired(z["tts"], pad["tts"]))
        d["zero_shot"][str(k)]["pad"] = per_sc(pad["tts"]) | {"unserved": int(pad["unserved"].sum())}
        d["zero_shot"][str(k)]["vs_scratch_ppo"] = {
            "scratch_mean": r2(np.mean(scratch_means)), "scratch_sd": r2(np.std(scratch_means, ddof=1)),
            "pct": r2(100 * (z["tts"].mean() / np.mean(scratch_means) - 1))}
        s = scored("transfer", f"net{k}")
        if s:
            d["zero_shot"][str(k)]["trainability"] = s["trainability"]
            d["zero_shot"][str(k)]["predictions"] = s["predictions"]
    zs = [load("L2W", f"transfer_net{k}")["tts"].mean() for k in NETS]
    d["zero_shot_vs_scratch_welch"] = ci(welch(zs, scratch_means))
    out["transfer"] = d


def patch(out):
    d = {"runs": {}, "pooled": {}, "baselines": {}, "writers": {}}
    T = {k: load("L2", target(k))["tts"] for k in NETS}
    for b in (0, 5, 10):
        pairs = []
        for k in NETS:
            x = load("L2", f"patch_net{k}_b{b}")
            s = scored("patch", f"net{k}_b{b}")
            ev = os.path.join(PACKS, "patch", f"net{k}_b{b}", "eval_log.jsonl")
            calls = [json.loads(l) for l in open(ev)] if os.path.exists(ev) else []
            d["runs"][f"net{k}_b{b}"] = {"vs_target": ci(paired(x["tts"], T[k])), "tts": per_sc(x["tts"]),
                                         "unserved": int(x["unserved"].sum()), "params_changed": s["params_changed"],
                                         "calls_used": len(calls), "dev_calls": [r2(c["tts_mean"]) for c in calls],
                                         "predictions": s["predictions"]}
            pairs.append((x["tts"], T[k]))
        d["pooled"][str(b)] = ci(pooled(pairs))
    for name, fmt in (("es_seed0", "es_net{k}_seed0"), ("es_seed1", "es_net{k}_seed1"), ("wrapper", "wrapper_net{k}"),
                      ("finetune_300k", "ft_net{k}_s60{k}_300k"), ("finetune_1M", "ft_net{k}_s60{k}_1000k")):
        d["baselines"][name] = {"pooled": ci(pooled([(load("L2", fmt.format(k=k))["tts"], T[k]) for k in NETS])),
                                "per_target": {str(k): ci(paired(load("L2", fmt.format(k=k))["tts"], T[k])) for k in NETS}}
    d["baselines"]["es_selection"] = json.load(open(os.path.join(V3, "results", "baselines", "es_selection.json")))
    p10 = [load("L2", f"patch_net{k}_b10")["tts"] for k in NETS]
    for name in ("es_seed0", "es_seed1", "finetune_1M"):
        fmt = {"es_seed0": "es_net{k}_seed0", "es_seed1": "es_net{k}_seed1", "finetune_1M": "ft_net{k}_s60{k}_1000k"}[name]
        d["baselines"][f"patch10_vs_{name}"] = ci(pooled([(p10[k], load("L2", fmt.format(k=k))["tts"]) for k in NETS]))
    for m in ("sonnet", "haiku", "fable"):
        rows = {}
        for k in NETS:
            x = load("L2", f"patchm_net{k}_{m}")
            s = scored("patch_models", f"net{k}_{m}")
            ev = os.path.join(PACKS, "patch_models", f"net{k}_{m}", "eval_log.jsonl")
            rows[str(k)] = {"vs_target": ci(paired(x["tts"], T[k])), "params_changed": s["params_changed"],
                            "calls_used": sum(1 for _ in open(ev)) if os.path.exists(ev) else 0,
                            "unserved": int(x["unserved"].sum())}
        d["writers"][m] = {"runs": rows, "pooled": ci(pooled([(load("L2", f"patchm_net{k}_{m}")["tts"], T[k]) for k in NETS]))}
    out["patch"] = d


def patch_fault(out):
    """Amendment P-F: Patch under a failed detector."""
    F = {k: load("L2_fault", f"faulty_net{k}")["tts"] for k in NETS}
    H = {k: load("L2", target(k))["tts"] for k in NETS}
    fault = json.load(open(os.path.join(V3, "results", "fault.json")))
    d = {"fault": fault, "screen": json.load(open(os.path.join(V3, "results", "fault_screen.json")))["screen"],
         "healthy": {str(k): per_sc(H[k]) for k in NETS},
         "faulty": {str(k): per_sc(F[k]) | {"unserved": int(load("L2_fault", f"faulty_net{k}")["unserved"].sum())} for k in NETS},
         "actuated_faulty": per_sc(load("L2_fault", "actuated")["tts"]), "fixed": per_sc(load("L2_fault", "fixed")["tts"]),
         "methods": {}}

    def method(name, fmt, extra=None):
        xs = {k: load("L2_fault", fmt.format(k=k)) for k in NETS}
        if any(v is None for v in xs.values()):
            return
        rows = {}
        for k in NETS:
            x = xs[k]["tts"]
            rows[str(k)] = {"vs_faulty": ci(paired(x, F[k])), "tts": per_sc(x), "unserved": int(xs[k]["unserved"].sum()),
                            "recovered_pct": r2(100 * (F[k].mean() - x.mean()) / (F[k].mean() - H[k].mean())),
                            "vs_healthy": ci(paired(x, H[k]))}
            if extra:
                rows[str(k)].update(extra(k))
        d["methods"][name] = {"runs": rows, "pooled_vs_faulty": ci(pooled([(xs[k]["tts"], F[k]) for k in NETS]))}

    for b in (0, 5, 10):
        def ex(k, b=b):
            s = scored("patch_fault", f"net{k}_b{b}")
            ev = os.path.join(PACKS, "patch_fault", f"net{k}_b{b}", "eval_log.jsonl")
            return {"params_changed": s["params_changed"], "calls_used": sum(1 for _ in open(ev)) if os.path.exists(ev) else 0,
                    "predictions": s["predictions"]}
        method(f"agents_b{b}", f"patchf_net{{k}}_b{b}", ex)
    method("mean_imputation", "meanimp_net{k}")
    method("es_seed0", "es_fault_net{k}_seed0")
    method("es_seed1", "es_fault_net{k}_seed1")
    method("finetune_300k", "ft_net{k}_s80{k}_300k")
    method("finetune_1M", "ft_net{k}_s80{k}_1000k")
    d["es_selection"] = json.load(open(os.path.join(V3, "results", "baselines", "es_fault_selection.json")))
    for b in (0, 5, 10):
        A = {k: load("L2_fault", f"patchf_net{k}_b{b}")["tts"] for k in NETS}
        for base, fmt in (("mean_imputation", "meanimp_net{k}"), ("es_seed0", "es_fault_net{k}_seed0"),
                          ("es_seed1", "es_fault_net{k}_seed1"), ("finetune_1M", "ft_net{k}_s80{k}_1000k")):
            B = {k: load("L2_fault", fmt.format(k=k))["tts"] for k in NETS}
            d[f"agents_b{b}_vs_{base}"] = {"pooled": ci(pooled([(A[k], B[k]) for k in NETS])),
                                           "per_target": {str(k): ci(paired(A[k], B[k])) for k in NETS}}
    out["patch_fault"] = d


def patch_fault_rep(out):
    """Amendment P-F-R: three sessions per network for the one-shot and 10-call arms (first run + replicates r2, r3).
    Descriptive per session; the H-F1/H-F2 comparisons repeated with each network's episode-wise mean over its
    sessions."""
    F = {k: load("L2_fault", f"faulty_net{k}")["tts"] for k in NETS}
    H = {k: load("L2", target(k))["tts"] for k in NETS}
    d = {"arms": {}}
    for b in (0, 10):
        arm = {"sessions": {}, "mean_of_sessions": {}}
        means = {}
        for k in NETS:
            labs = [f"patchf_net{k}_b{b}"] + [f"patchf_net{k}_b{b}_r{r}" for r in (2, 3)]
            xs = [load("L2_fault", l) for l in labs]
            if any(x is None for x in xs):
                return
            rows = []
            for l, x in zip(labs, xs):
                t = x["tts"]
                pid = l.replace("patchf_", "")
                s_ = scored("patch_fault", pid)
                ev = os.path.join(PACKS, "patch_fault", pid, "eval_log.jsonl")
                rows.append({"label": l, "tts": r2(t.mean()),
                             "recovered_pct": r2(100 * (F[k].mean() - t.mean()) / (F[k].mean() - H[k].mean())),
                             "vs_faulty": ci(paired(t, F[k])), "vs_healthy": ci(paired(t, H[k])),
                             "unserved": int(x["unserved"].sum()), "params_changed": s_["params_changed"] if s_ else None,
                             "calls_used": sum(1 for _ in open(ev)) if os.path.exists(ev) else 0})
            arm["sessions"][str(k)] = rows
            rec = [r["recovered_pct"] for r in rows]
            means[k] = np.mean([x["tts"] for x in xs], axis=0)
            arm["mean_of_sessions"][str(k)] = {
                "recovered_mean": r2(np.mean(rec)), "recovered_min": r2(min(rec)), "recovered_max": r2(max(rec)),
                "recovered_of_mean_tts": r2(100 * (F[k].mean() - means[k].mean()) / (F[k].mean() - H[k].mean())),
                "sd_session_tts": r2(np.std([x["tts"].mean() for x in xs], ddof=1)),
                "vs_faulty": ci(paired(means[k], F[k])), "vs_healthy": ci(paired(means[k], H[k]))}
        arm["pooled_vs_faulty"] = ci(pooled([(means[k], F[k]) for k in NETS]))
        for base, fmt in (("mean_imputation", "meanimp_net{k}"), ("es_seed0", "es_fault_net{k}_seed0"),
                          ("es_seed1", "es_fault_net{k}_seed1"), ("finetune_1M", "ft_net{k}_s80{k}_1000k")):
            B = {k: load("L2_fault", fmt.format(k=k))["tts"] for k in NETS}
            arm[f"vs_{base}"] = {"pooled": ci(pooled([(means[k], B[k]) for k in NETS])),
                                 "per_target": {str(k): ci(paired(means[k], B[k])) for k in NETS}}
        d["arms"][f"b{b}"] = arm
    out["patch_fault_rep"] = d


def explain(out, do_eval):
    from wsp.agents_eval import comparisons, fidelity, heldout_states
    from wsp.policies import load_net
    d = {"agents": {}, "trees": {}}
    for k in NETS:
        net = load("L2", target(k))["tts"]
        s = scored("explain", f"net{k}")
        X = heldout_states(k)
        w = load_net(os.path.join(V3, "models", "L2", target(k), "final.zip"))
        row = {}
        for n in ("small", "full"):
            p = os.path.join(PACKS, "explain", f"net{k}", "work", f"rules_{n}.py")
            x = load("L2", f"explain_net{k}_{n}")
            row[n] = {"comparisons": s[f"comparisons_{n}"], "fidelity": r2(s[f"fidelity_{n}"]),
                      "constants": count_constants(p), "vs_network": ci(paired(x["tts"], net)),
                      "unserved": int(x["unserved"].sum())}
        row["predictions"] = s["predictions"]
        d["agents"][str(k)] = row
        for kind in ("cart", "dagger"):
            for n in ("small", "mid", "full"):
                p = os.path.join(V3, "models", "L2", "baselines", f"{kind}_net{k}_{n}.py")
                lab = f"{kind}_net{k}_{n}"
                if do_eval:
                    evaluate("L2", f"rules:{p}", "test", lab, workers=10)
                x = load("L2", lab)
                d["trees"][lab] = {"comparisons": comparisons(p), "constants": count_constants(p),
                                   "fidelity": r2(fidelity(p, w, X)),
                                   "vs_network": ci(paired(x["tts"], net)) if x is not None else None,
                                   "unserved": int(x["unserved"].sum()) if x is not None else None}
    out["explain"] = d


def count_constants(path):
    import ast
    tree = ast.parse(open(path).read())
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
               and not isinstance(n.value, bool))


def constrain(out):
    d = {"runs": {}, "pooled": {}, "baselines": {}}
    N = {k: load("L2_rules", f"naive_net{k}")["tts"] for k in NETS}
    U = {k: load("L2", target(k))["tts"] for k in NETS}
    d["rules_cost"] = ci(pooled([(N[k], U[k]) for k in NETS]))
    for b in (0, 10):
        pairs = []
        for k in NETS:
            x = load("L2_rules", f"constrain_net{k}_b{b}")
            ev = os.path.join(PACKS, "constrain", f"net{k}_b{b}", "eval_log.jsonl")
            d["runs"][f"net{k}_b{b}"] = {"vs_naive": ci(paired(x["tts"], N[k])), "vs_unconstrained": ci(paired(x["tts"], U[k])),
                                         "tts": per_sc(x["tts"]),
                                         "calls_used": sum(1 for _ in open(ev)) if os.path.exists(ev) else 0,
                                         "predictions": (scored("constrain", f"net{k}_b{b}") or {}).get("predictions")}
            pairs.append((x["tts"], N[k]))
        d["pooled"][str(b)] = ci(pooled(pairs))
    for name, fmt in (("masked", "masked_net{k}"), ("finetune_300k", "ft_net{k}_s70{k}_300k"),
                      ("finetune_1M", "ft_net{k}_s70{k}_1000k")):
        d["baselines"][name] = ci(pooled([(load("L2_rules", fmt.format(k=k))["tts"], N[k]) for k in NETS]))
    scratch = [load("L2_rules", f"ppo_s{s}")["tts"] for s in range(300, 303)]
    d["baselines"]["scratch_under_rules_vs_naive"] = ci(pooled([(scratch[k], N[k]) for k in NETS]))
    d["baselines"]["actuated_under_rules"] = per_sc(load("L2_rules", "actuated")["tts"])
    d["baselines"]["fixed_under_rules"] = per_sc(load("L2_rules", "fixed")["tts"])
    loop = [load("L2_rules", f"constrain_net{k}_b10")["tts"] for k in NETS]
    d["loop_vs_finetune_1M"] = ci(pooled([(loop[k], load("L2_rules", f"ft_net{k}_s70{k}_1000k")["tts"]) for k in NETS]))
    d["loop_vs_scratch_welch"] = ci(welch([x.mean() for x in loop], [x.mean() for x in scratch]))
    out["constrain"] = d


def calibration(out):
    rows = []
    for case, pid, prob, lab in ([("init", f"r{k}", "L2", f"init_r{k}") for k in (1, 2, 3)]
                                 + [("patch", f"net{k}_b{b}", "L2", f"patch_net{k}_b{b}") for k in NETS for b in (0, 5, 10)]
                                 + [("patch_fault", f"net{k}_b{b}", "L2_fault", f"patchf_net{k}_b{b}") for k in NETS for b in (0, 5, 10)]
                                 + [("transfer", f"net{k}", "L2W", f"transfer_net{k}") for k in NETS]
                                 + [("constrain", f"net{k}_b{b}", "L2_rules", f"constrain_net{k}_b{b}") for k in NETS for b in (0, 10)]):
        s = scored(case, pid)
        x = load(prob, lab)
        if not s or not s.get("predictions") or x is None:
            continue
        pr = s["predictions"]
        err = {sc: r2(100 * (float(pr[sc]) / x["tts"][sl].mean() - 1)) for sc, sl in SC.items() if sc in pr}
        rows.append({"case": case, "run": pid, "signed_error_pct": err})
    out["calibration"] = rows


def report(out):
    R = out["reference"]
    L = ["# v3 results (30 held-out test episodes; % = change in mean TTS, 95 % CI)\n"]
    for p in ("L2", "L2W", "L2_rules"):
        r = R[p]
        L.append(f"**{p}**: fixed {r['fixed']['all']:.2f}, actuated {r['actuated']['all']:.2f}, PPO {r['ppo_mean']['all']:.2f} "
                 f"(sd over seeds {r['ppo_sd_runs']:.2f}); PPO vs actuated {r['ppo_vs_actuated']['pct']:+.1f} % "
                 f"[{r['ppo_vs_actuated']['pct_lo']:+.1f}, {r['ppo_vs_actuated']['pct_hi']:+.1f}], vs fixed "
                 f"{r['ppo_vs_fixed']['pct']:+.1f} %\n")
    nc = [k for k, v in R["convergence"].items() if v["converged"] is False]
    sup = {k: f"{k}_r1" for k in nc if f"{k}_r1" in R["convergence"]}
    L.append(f"Runs failing the convergence criterion: {nc or 'none'}"
             + "".join(f"; {k} is superseded by {v} (converged: {R['convergence'][v]['converged']}, "
                       f"{R['convergence'][v]['steps'] / 1e6:.1f}M steps), which the analysis uses" for k, v in sup.items()) + "\n")
    for name in ("initialise", "transfer"):
        d = out[name]
        L.append(f"\n## {name.capitalize()}\n")
        for k, z in d["zero_shot"].items():
            L.append(f"- zero-shot {k}: TTS {z['tts']['all']:.2f}; vs fixed {z['vs_fixed']['pct']:+.1f} %, vs actuated "
                     f"{z['vs_actuated']['pct']:+.1f} %" + (f", vs padding {z['vs_pad']['pct']:+.1f} %" if "vs_pad" in z else ""))
        for key in ("first50_mean_delay", "first1M_total_delay", "final_tts"):
            r = d.get(f"warm_vs_scratch_{key}")
            if r:
                L.append(f"- warm vs scratch, {key}: {r['mean_a']:.1f} vs {r['mean_b']:.1f} ({r['pct']:+.1f} % "
                         f"[{r['pct_lo']:+.1f}, {r['pct_hi']:+.1f}])")
    d = out["patch"]
    L.append("\n## Patch (vs each target; pooled over 3 targets)\n")
    for b, r in d["pooled"].items():
        L.append(f"- budget {b}: {r['pct']:+.2f} % [{r['pct_lo']:+.2f}, {r['pct_hi']:+.2f}], wins {r['wins']}/30")
    for b, r in d["baselines"].items():
        if isinstance(r, dict) and "pooled" in r:
            L.append(f"- {b}: {r['pooled']['pct']:+.2f} % [{r['pooled']['pct_lo']:+.2f}, {r['pooled']['pct_hi']:+.2f}]")
    for m, r in d["writers"].items():
        L.append(f"- writer {m}: {r['pooled']['pct']:+.2f} % [{r['pooled']['pct_lo']:+.2f}, {r['pooled']['pct_hi']:+.2f}]")
    d = out["patch_fault"]
    L.append(f"\n## Patch under a failed detector ({d['fault']['name']}; recovered = share of the fault's cost)\n")
    for k in NETS:
        L.append(f"- net{k}: healthy {d['healthy'][str(k)]['all']:.1f}, faulty {d['faulty'][str(k)]['all']:.1f}")
    for name, m in d["methods"].items():
        L.append(f"- {name}: pooled vs faulty {m['pooled_vs_faulty']['pct']:+.1f} %; recovered "
                 + " / ".join(f"{m['runs'][str(k)]['recovered_pct']:.0f} %" for k in NETS))
    for b in (0, 5, 10):
        for base in ("mean_imputation", "es_seed0", "finetune_1M"):
            r = d[f"agents_b{b}_vs_{base}"]
            L.append(f"- agents b{b} vs {base}: pooled {r['pooled']['pct']:+.1f} % [{r['pooled']['pct_lo']:+.1f}, "
                     f"{r['pooled']['pct_hi']:+.1f}]; per target "
                     + ", ".join(f"{v['pct']:+.1f} % [{v['pct_lo']:+.1f}, {v['pct_hi']:+.1f}]" for v in r["per_target"].values()))
    if "patch_fault_rep" in out:
        L.append("\n## Failed detector: three sessions per network (first run + 2 replicates)\n")
        for b, arm in out["patch_fault_rep"]["arms"].items():
            for k, m in arm["mean_of_sessions"].items():
                ss = arm["sessions"][k]
                L.append(f"- {b} net{k}: recovered " + " / ".join(f"{r['recovered_pct']:.0f} %" for r in ss)
                         + f" (mean {m['recovered_mean']:.0f} %); session-mean vs faulty {m['vs_faulty']['pct']:+.1f} %")
            for base in ("mean_imputation", "es_seed0", "es_seed1", "finetune_1M"):
                r = arm[f"vs_{base}"]
                L.append(f"- {b} session-mean vs {base}: pooled {r['pooled']['pct']:+.1f} % [{r['pooled']['pct_lo']:+.1f}, "
                         f"{r['pooled']['pct_hi']:+.1f}]; per target " + ", ".join(f"{v['pct']:+.1f}{'*' if v['sig'] else ''}"
                                                                                   for v in r["per_target"].values()))
    d = out["explain"]
    L.append("\n## Explain (fidelity on held-out honoured states; closed-loop TTS vs the network)\n")
    for k, a in d["agents"].items():
        for n in ("small", "full"):
            x = a[n]
            L.append(f"- agent net{k} {n}: {x['comparisons']} comparisons, {x['constants']} constants, fidelity "
                     f"{100 * x['fidelity']:.1f} %, TTS {x['vs_network']['pct']:+.1f} %")
    for lab, x in d["trees"].items():
        v = x["vs_network"]
        L.append(f"- {lab}: {x['comparisons']} comparisons, fidelity {100 * x['fidelity']:.1f} %"
                 + (f", TTS {v['pct']:+.1f} %, unserved {x['unserved']}" if v else ""))
    d = out["constrain"]
    L.append(f"\n## Constrain (cost of the rules under the naive shield: {d['rules_cost']['pct']:+.2f} %)\n")
    for b, r in d["pooled"].items():
        L.append(f"- supervisors, budget {b}: vs naive {r['pct']:+.2f} % [{r['pct_lo']:+.2f}, {r['pct_hi']:+.2f}]")
    for b, r in d["baselines"].items():
        if isinstance(r, dict) and "pct" in r:
            L.append(f"- {b}: {r['pct']:+.2f} % [{r['pct_lo']:+.2f}, {r['pct_hi']:+.2f}]")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--cap", type=int, default=1800,
                    help="drain cap (s) of the test evaluations to use; 3600 = the robustness check of the appendix")
    a = ap.parse_args()
    suffix = "" if a.cap == 1800 else f"_cap{a.cap}"
    if a.cap != 1800:
        # an evaluation with no unserved vehicles at 30 min ended before the cap, so it is identical under any longer cap
        import wsp.stats as S
        orig = S.load

        def load_cap(problem, label, set_name="test", drain_cap=1800):
            x = orig(problem, label, set_name, a.cap)
            if x is not None:
                return x
            x0 = orig(problem, label, set_name, 1800)
            if x0 is not None and x0["unserved"].sum() > 0:
                raise RuntimeError(f"{problem}/{label}: unserved vehicles at 30 min but no {a.cap}-s evaluation")
            return x0
        globals()["load"] = load_cap
    out = {"drain_cap_s": a.cap}
    reference(out)
    initialise(out)
    transfer(out)
    patch(out)
    patch_fault(out)
    patch_fault_rep(out)
    explain(out, a.evaluate)
    constrain(out)
    calibration(out)
    os.makedirs(os.path.join(V3, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(V3, "results", f"paper_numbers{suffix}.json"), "w"), indent=1, default=float)
    txt = report(out)
    open(os.path.join(V3, "results", f"RESULTS{suffix}.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
