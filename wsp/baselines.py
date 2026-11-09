# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Non-LLM baselines of the agent studies (docs/EXPERIMENT_PLAN.md), with the agents' budgets where they apply.

    (from the repository root)
    python -m wsp.baselines es       --targets 0,1,2     # Patch: (1+1)-ES on the output layer, 10 dev evaluations
    python -m wsp.baselines wrapper  --targets 0,1,2     # Patch, exploratory: execution-time hold rule, 10 dev evals
    python -m wsp.baselines pad      --targets 0,1,2     # Transfer: positional padding L2 -> L2W
    python -m wsp.baselines cart     --targets 0,1,2     # Explain: CART trees (depth 3, 5, 8) on the pack's states
    python -m wsp.baselines dagger   --targets 0,1,2     # Explain: DAgger-style iterative distillation

Targets k are the L2 PPO networks Net k = models/L2/ppo_s{100 + k}/final.zip. Every candidate evaluation covers the
9 dev episodes, as for the closed-loop agents, and candidates are ranked by mean dev TTS (which already counts
vehicles left at the cap). Outputs: networks / rule files in models/<problem>/baselines/, logs in
results/baselines/*.json.

es       (1+1)-ES with the 1/5 success rule on W3, b3 (260 parameters, within the 5 % cap): additive Gaussian steps
         of sigma * std(W3 of the target); x1.5 after a success, x1.5^(-1/4) after a failure; 10 candidate
         evaluations (the target's own dev TTS is known, as it is to the agents); the best network evaluated is
         submitted (the target if nothing improves). Grid sigma0 in {0.03, 0.1, 0.3} x ES seeds {0, 1}; the sigma0
         with the best mean val TTS of the submitted networks over all targets and seeds is selected.
wrapper  wsp.policies.HoldWrapper with k in {1, 2, 4, 6, 8} halting vehicles x max_hold in {30, 45} s, each evaluated
         once on dev; the best is kept. (Exploratory: the rule the closed-loop agents implement may differ, in which
         case the agents' own rule is evaluated the same way in the analysis.)
pad      W1 of the L2W network = [W1 of the L2 network, zeros for the 8 extra inputs] (columns copied by index);
         b1, W2, b2, W3, b3 unchanged.
cart     wsp.tree (Gini, min leaf 10, <= 64 quantile thresholds per feature) on the honoured states of the Explain
         pack's states.npz, labelled with the network's greedy phase; depths 3, 5, 8.
dagger   D_0 = the cart data; for i = 0..4 roll tree_i out on the 6 rec episodes, label the honoured states it visits
         with the network's greedy phase, D_{i+1} = D_i + those states, refit; tree_5 is rolled out too. The
         iteration with the lowest mean rollout TTS is selected (no extra simulation).
"""
import argparse
import json
import os

import numpy as np

from wsp import tree as T
from wsp.env import SignalEnv
from wsp.evaluate import SETS, V3, evaluate, run_many, jobs_for
from wsp.policies import forward, load_net

OUT = os.path.join(V3, "results", "baselines")
SIGMAS = (0.03, 0.1, 0.3)
ES_SEEDS = (0, 1)
BUDGET = 10
WRAP_GRID = [(k, mh) for mh in (30, 45) for k in (1, 2, 4, 6, 8)]
DEPTHS = (("small", 3), ("mid", 5), ("full", 8))
ITERS = 5


def target(k, problem="L2"):
    return os.path.join(V3, "models", problem, f"ppo_s{100 + k}", "final.zip")


def nets_dir(problem="L2"):
    d = os.path.join(V3, "models", problem, "baselines")
    os.makedirs(d, exist_ok=True)
    return d


def dev(problem, policy, label):
    rows = run_many(jobs_for(problem, policy, "dev", label), 9)
    return {"tts": float(np.mean([r["tts_veh_h"] for r in rows])), "unserved": int(sum(r["remaining"] for r in rows)),
            "tts_per_episode": [r["tts_veh_h"] for r in rows]}


def save_net(w, path):
    np.savez(path, **{k: np.asarray(v, np.float32) for k, v in w.items()})
    return path


# ---------------------------------------------------------------- Patch
def es_run(k, sigma0, seed, problem="L2"):
    tag = "" if problem == "L2" else "fault_"
    path = os.path.join(OUT, f"es_{tag}net{k}_sig{sigma0}_seed{seed}.json")
    if os.path.exists(path):
        return json.load(open(path))
    w0 = load_net(target(k))
    theta0 = np.r_[w0["W3"].ravel(), w0["b3"]]
    scale = float(w0["W3"].std())
    rng = np.random.default_rng(1000 * seed + int(round(sigma0 * 1000)))
    work = os.path.join(V3, "runs", "baselines", f"es_{tag}net{k}_sig{sigma0}_seed{seed}")
    os.makedirs(work, exist_ok=True)

    def net(theta):
        w = {kk: v.copy() for kk, v in w0.items()}
        w["W3"] = theta[:w0["W3"].size].reshape(w0["W3"].shape)
        w["b3"] = theta[w0["W3"].size:]
        return w
    start = dev(problem, f"sb3:{target(k)}", f"bl_{tag}start_net{k}")
    best_theta, best, sigma, log = theta0.copy(), dict(start, call=0), sigma0, []
    for i in range(1, BUDGET + 1):
        cand = best_theta + sigma * scale * rng.standard_normal(theta0.size)
        p = save_net(net(cand), os.path.join(work, f"cand_{i:02d}.npz"))
        r = dev(problem, f"npz:{p}", f"bl_es_{tag}net{k}_{sigma0}_{seed}_{i}")
        ok = r["tts"] < best["tts"]
        log.append({"call": i, "sigma": sigma, "accepted": bool(ok), **r})
        if ok:
            best_theta, best, sigma = cand, dict(r, call=i), sigma * 1.5
        else:
            sigma *= 1.5 ** -0.25
    sub = save_net(net(best_theta), os.path.join(nets_dir(), f"es_{tag}net{k}_sig{sigma0}_seed{seed}.npz"))
    val = evaluate(problem, f"npz:{sub}", "val", f"es_{tag}net{k}_sig{sigma0}_seed{seed}", workers=6)
    out = {"target": k, "sigma0": sigma0, "seed": seed, "scale": scale, "start": start, "log": log,
           "submitted_call": best["call"], "submitted_dev": best["tts"], "network": sub,
           "params_changed": int(np.sum(best_theta != theta0)), "val_tts": val["summary"]["all"]["tts_veh_h"]}
    json.dump(out, open(path, "w"), indent=1)
    print(f"ES net{k} sigma0={sigma0} seed={seed}: accepted {sum(l['accepted'] for l in log)}/10, dev "
          f"{start['tts']:.2f} -> {best['tts']:.2f}, val {out['val_tts']:.2f}", flush=True)
    return out


def stage_es(targets, problem="L2"):
    tag = "" if problem == "L2" else "fault_"
    res = {s: [es_run(k, s, seed, problem) for k in targets for seed in ES_SEEDS] for s in SIGMAS}
    mean = {s: float(np.mean([r["val_tts"] for r in v])) for s, v in res.items()}
    sel = min(mean, key=mean.get)
    json.dump({"targets": targets, "problem": problem, "val_mean_by_sigma": {str(s): m for s, m in mean.items()},
               "selected_sigma0": sel}, open(os.path.join(OUT, f"es_{tag}selection.json"), "w"), indent=1)
    for k in targets:
        for seed in ES_SEEDS:
            r = es_run(k, sel, seed, problem)
            evaluate(problem, f"npz:{r['network']}", "test", f"es_{tag}net{k}_seed{seed}", workers=10)
    print("ES", problem, ": selected sigma0", sel, mean)


def stage_es_fault(targets):
    stage_es(targets, "L2_fault")


def stage_meanimp(targets):
    """Failed-detector study: mean imputation folded into b1 (the dead inputs replaced by their mean over the
    target's own recorded rec states, with every detector working)."""
    from wsp.env import SignalEnv
    env = SignalEnv("L2_fault", workdir=os.path.join(V3, "runs", "tmp", "mi"))
    idx = env._zero_idx
    for k in targets:
        lab = f"target_net{k}"
        evaluate("L2", f"sb3:{target(k)}", "rec", lab, workers=6, record=True)
        X = np.concatenate([np.load(os.path.join(V3, "runs", "eval", "L2", lab, "rec", f"{sc}_{s}", "record.npz"))["obs"]
                            for sc, s in SETS["rec"]]).astype(np.float64)
        w = load_net(target(k))
        mu = X[:, idx].mean(axis=0)
        w["b1"] = w["b1"] + w["W1"][:, idx] @ mu
        p = save_net(w, os.path.join(nets_dir(), f"meanimp_net{k}.npz"))
        r = evaluate("L2_fault", f"npz:{p}", "test", f"meanimp_net{k}", workers=10)
        print(f"mean imputation net{k}: inputs {idx} means {np.round(mu, 3)}; test TTS {r['summary']['all']['tts_veh_h']:.2f}",
              flush=True)


def stage_wrapper(targets):
    for k in targets:
        path = os.path.join(OUT, f"wrapper_net{k}.json")
        if os.path.exists(path):
            continue
        log = []
        for i, (kk, mh) in enumerate(WRAP_GRID):
            r = dev("L2", f"hold:{kk},{mh}|{target(k)}", f"bl_wrap_net{k}_{i}")
            log.append({"call": i + 1, "k": kk, "max_hold": mh, **r})
        best = min(log, key=lambda r: r["tts"])
        json.dump({"target": k, "grid": log, "selected": {"k": best["k"], "max_hold": best["max_hold"]}},
                  open(path, "w"), indent=1)
        evaluate("L2", f"hold:{best['k']},{best['max_hold']}|{target(k)}", "test", f"wrapper_net{k}", workers=10)
        print(f"wrapper net{k}: k={best['k']} max_hold={best['max_hold']} dev {best['tts']:.2f}", flush=True)


# ---------------------------------------------------------------- Transfer
def stage_pad(targets):
    n_b = SignalEnv("L2W", workdir=os.path.join(V3, "runs", "tmp", "pad")).observation_space.shape[0]
    for k in targets:
        w = load_net(target(k))
        W1 = np.zeros((w["W1"].shape[0], n_b))
        W1[:, :w["W1"].shape[1]] = w["W1"]
        p = save_net(dict(w, W1=W1), os.path.join(nets_dir("L2W"), f"pad_net{k}.npz"))
        r = evaluate("L2W", f"npz:{p}", "test", f"pad_net{k}", workers=10)
        print(f"pad net{k}: test TTS {r['summary']['all']['tts_veh_h']:.2f}", flush=True)


# ---------------------------------------------------------------- Explain
def explain_states(k):
    S = np.load(os.path.join(V3, "packs", "explain", f"net{k}", "states.npz"))
    X = S["obs"].astype(np.float64)
    ok = SignalEnv("L2", workdir=os.path.join(V3, "runs", "tmp", "tree")).ok_index()
    return X[X[:, ok] > 0.5], ok


def greedy(w, X):
    return forward(w, X).argmax(axis=1)


def names():
    return SignalEnv("L2", workdir=os.path.join(V3, "runs", "tmp", "tree")).obs_names()


def stage_cart(targets):
    for k in targets:
        w = load_net(target(k))
        X, _ = explain_states(k)
        y = greedy(w, X)
        for name, depth in DEPTHS:
            tr = T.fit(X, y, max_depth=depth, min_leaf=10)
            p = os.path.join(nets_dir(), f"cart_net{k}_{name}.py")
            open(p, "w").write(T.to_python(tr, names(), header=f"CART distillation of Net {k}, depth {depth}"))
            print(f"cart net{k} {name}: {T.n_splits(tr)} comparisons", flush=True)


def stage_dagger(targets):
    log_path = os.path.join(OUT, "dagger.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    work = os.path.join(V3, "runs", "baselines", "dagger")
    os.makedirs(work, exist_ok=True)
    for k in targets:
        w = load_net(target(k))
        X0, ok = explain_states(k)
        for name, depth in DEPTHS:
            key = f"net{k}_{name}"
            if key in log:
                continue
            X, its = X0.copy(), []
            for i in range(ITERS + 1):
                y = greedy(w, X)
                tr = T.fit(X, y, max_depth=depth, min_leaf=10)
                path = os.path.join(work, f"{key}_iter{i}.py")
                open(path, "w").write(T.to_python(tr, names(), header=(
                    f"DAgger distillation of Net {k}, depth {depth}, iteration {i}; {len(X)} labelled states")))
                lab = f"dagger_{key}_{i}"
                d = evaluate("L2", f"rules:{path}", "rec", lab, workers=6, record=True)
                Xr = np.concatenate([np.load(os.path.join(V3, "runs", "eval", "L2", lab, "rec", f"{sc}_{s}",
                                                          "record.npz"))["obs"] for sc, s in SETS["rec"]]).astype(np.float64)
                its.append({"iter": i, "n_states": int(len(X)), "comparisons": T.n_splits(tr),
                            "train_agreement": float((T.predict(tr, X) == y).mean()),
                            "rollout_tts": d["summary"]["all"]["tts_veh_h"], "path": path})
                X = np.concatenate([X, Xr[Xr[:, ok] > 0.5]])
            best = min(its, key=lambda r: r["rollout_tts"])
            out = os.path.join(nets_dir(), f"dagger_net{k}_{name}.py")
            open(out, "w").write(open(best["path"]).read().replace(
                '"""DAgger', f'"""Selected (iteration {best["iter"]} of 0-{ITERS}). DAgger', 1))
            log[key] = {"iterations": its, "selected_iter": best["iter"], "file": out}
            json.dump(log, open(log_path, "w"), indent=1)
            print(f"dagger {key}: selected iteration {best['iter']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["es", "es_fault", "meanimp", "wrapper", "pad", "cart", "dagger"])
    ap.add_argument("--targets", default="0,1,2")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.makedirs(OUT, exist_ok=True)
    t = [int(x) for x in a.targets.split(",")]
    {"es": stage_es, "es_fault": stage_es_fault, "meanimp": stage_meanimp, "wrapper": stage_wrapper, "pad": stage_pad,
     "cart": stage_cart, "dagger": stage_dagger}[a.stage](t)


if __name__ == "__main__":
    main()
