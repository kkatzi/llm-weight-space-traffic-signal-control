# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Konstantinos Katzilieris, Emmanouil Kampitakis, Eleni I. Vlahogianni (NTUA)
"""Score a finished agent run: check its deliverables against the task's rules, evaluate them on the held-out test
episodes and (Initialise, Transfer) queue the follow-on PPO runs.

    (from the repository root)  python -m wsp.agents_eval packs/patch/net0_b10 [packs/... ...]

Writes <pack>/scored.json and the test evaluations (results/eval/<problem>/<label>/test.json).
    init       trainability on the pack's states (required bounds); zero-shot test on L2 (label init_r<k>);
               queues PPO warm starts: 3M steps, seeds 400-401, critic warm-up 50k (models/L2/init_r<k>_s<seed>)
    patch      parameters changed vs the target (cap 371); test on L2 (label patch_<id>, patchm_<id> for other writers)
    transfer   W2, b2, W3, b3 equal to the source; test on L2W (label transfer_net<k>); queues PPO warm starts on L2W:
               3M steps, seeds 500-501, critic warm-up 50k (models/L2W/transfer_net<k>_s<seed>)
    explain    comparisons per rule file (rules_small <= 8); fidelity on the held-out states (the target's own runs on
               the 6 val episodes, honoured steps only); test of each rule file as the controller
               (labels explain_net<k>_small / _full)
    constrain  test under the rules of supervisor + network (label constrain_<id>); the agreement with the network's
               own choice is computed in the analysis
"""
import ast
import json
import os
import sys

import numpy as np

from wsp.env import SignalEnv
from wsp.evaluate import SETS, V3, evaluate
from wsp.policies import forward, load_net

TRAIN_BOUNDS = {"median_max_prob": 0.95, "share_max_prob_above_0.99": 0.30, "max_abs_logit": 15.0}


def target(k, problem="L2"):
    return os.path.join(V3, "models", problem, f"ppo_s{100 + k}", "final.zip")


def edit_size(w, w0):
    return int(sum((np.abs(np.asarray(w[k]) - np.asarray(w0[k])) > 0).sum() for k in w0))


def trainability(w, X):
    from wsp.packs import trainability_of
    return trainability_of(w, X)


def comparisons(path):
    tree = ast.parse(open(path).read())
    return sum(len(n.ops) for n in ast.walk(tree) if isinstance(n, ast.Compare))


def heldout_states(k):
    """The target network's own trajectories on the 6 val episodes (recorded once), honoured steps only."""
    lab = f"heldout_net{k}"
    evaluate("L2", f"sb3:{target(k)}", "val", lab, workers=6, record=True)
    X = np.concatenate([np.load(os.path.join(V3, "runs", "eval", "L2", lab, "val", f"{sc}_{s}", "record.npz"))["obs"]
                        for sc, s in SETS["val"]]).astype(np.float64)
    ok = SignalEnv("L2", workdir=os.path.join(V3, "runs", "tmp", "ae")).ok_index()
    return X[X[:, ok] > 0.5]


def fidelity(rules_path, w, X):
    from wsp.policies import RulesPolicy
    dec = RulesPolicy(rules_path).decide
    net = forward(w, X).argmax(axis=1)
    rul = np.array([int(dec(x)) for x in X])
    return float((net == rul).mean())


def queue_warm_starts(problem, actor, prefix, seeds):
    jobs = []
    for s in seeds:
        name = f"{prefix}_s{s}"
        tid = f"train/{problem}/{name}/ind"
        jobs.append(dict(id=tid, kind="train", cores=4.5, steps=3_000_000, run_dir=f"models/{problem}/{name}",
                         stage="E-warmstarts", prio=7,
                         cmd=f"python -m wsp.train --problem {problem} --name {name} --seed {s} --steps 3000000 "
                             f"--ckpt-every 50000 --init-actor {actor} --critic-warmup 50000"))
        jobs.append(dict(id=f"curve/{problem}/{name}/ind", kind="eval", cores=4, stage="E-warmstarts", prio=9, after=[tid],
                         cmd=f"python -m wsp.curves --run models/{problem}/{name} --workers 4"))
        jobs.append(dict(id=f"test/{problem}/{name}/ind", kind="eval", cores=10, stage="E-warmstarts", prio=6,
                         after=[f"curve/{problem}/{name}/ind"],
                         cmd=f"python -m wsp.evaluate --problem {problem} --policy 'sb3:{V3}/models/{problem}/{name}/final.zip' "
                             f"--set test --label {name} --workers 10"))
    path = os.path.join(V3, "queue", "pending_add.jsonl")
    with open(path, "w") as f:
        f.write("".join(json.dumps(j) + "\n" for j in jobs))
    from wsp.queue import add
    add(path)


def score(pack):
    pack = os.path.abspath(pack)
    spec = json.load(open(os.path.join(pack, "spec.json")))
    case, pid = spec["case"], os.path.basename(pack)
    work = os.path.join(pack, "work")
    out = {"pack": pack, "case": case, "missing": []}
    use = os.path.basename(os.path.dirname(pack))
    pred = os.path.join(work, "predictions.json")
    out["predictions"] = json.load(open(pred)) if os.path.exists(pred) else None
    if case in ("init", "patch", "transfer", "patch_fault"):
        wpath = os.path.join(work, "weights.npz")
        if not os.path.exists(wpath):
            out["missing"].append("weights.npz")
            return out
        w = load_net(wpath)
        S = np.load(os.path.join(pack, "states.npz"))
        X = S["obs"].astype(np.float64)
        out["trainability"] = trainability(w, X)
        if case == "init":
            k = int(pid[1:])
            out["meets_trainability"] = all(out["trainability"][m] <= b for m, b in TRAIN_BOUNDS.items())
            r = evaluate("L2", f"npz:{wpath}", "test", f"init_r{k}", workers=10)
            if out["meets_trainability"] and not os.path.exists(os.path.join(V3, "models", "L2", f"init_r{k}_s400")):
                queue_warm_starts("L2", wpath, f"init_r{k}", (400, 401))
        elif case == "patch_fault":
            k = int(pid[3])
            out["params_changed"] = edit_size(w, load_net(target(k)))
            out["within_cap"] = out["params_changed"] <= 371
            r = evaluate("L2_fault", f"npz:{wpath}", "test", f"patchf_{pid}", workers=10)
        elif case == "patch":
            k = int(pid[3])
            w0 = load_net(target(k))
            out["params_changed"] = edit_size(w, w0)
            out["within_cap"] = out["params_changed"] <= 371
            label = ("patchm_" if use == "patch_models" else "patch_") + pid
            r = evaluate("L2", f"npz:{wpath}", "test", label, workers=10)
        else:
            k = int(pid[3])
            w0 = load_net(target(k))
            out["frozen_layers_equal"] = all(np.array_equal(np.asarray(w[n], np.float32), np.asarray(w0[n], np.float32))
                                             for n in ("W2", "b2", "W3", "b3"))
            r = evaluate("L2W", f"npz:{wpath}", "test", f"transfer_net{k}", workers=10)
            if out["frozen_layers_equal"] and not os.path.exists(os.path.join(V3, "models", "L2W", f"transfer_net{k}_s500")):
                queue_warm_starts("L2W", wpath, f"transfer_net{k}", (500, 501))
        out["test"] = r["summary"]
    elif case == "explain":
        k = int(pid[3])
        w0 = load_net(target(k))
        X = heldout_states(k)
        out["heldout_states"] = int(len(X))
        for name in ("small", "full"):
            p = os.path.join(work, f"rules_{name}.py")
            if not os.path.exists(p):
                out["missing"].append(f"rules_{name}.py")
                continue
            out[f"comparisons_{name}"] = comparisons(p)
            out[f"fidelity_{name}"] = fidelity(p, w0, X)
            out[f"test_{name}"] = evaluate("L2", f"rules:{p}", "test", f"explain_net{k}_{name}", workers=10)["summary"]
        out["within_size"] = out.get("comparisons_small", 99) <= 8
    elif case == "constrain":
        k = int(pid[3])
        sup = os.path.join(work, "supervisor.py")
        if not os.path.exists(sup):
            out["missing"].append("supervisor.py")
            return out
        lab = f"constrain_{pid}"
        out["test"] = evaluate("L2_rules", f"super:{sup}|{target(k)}", "test", lab, workers=10)["summary"]
    json.dump(out, open(os.path.join(pack, "scored.json"), "w"), indent=1, default=float)
    brief = {k: v for k, v in out.items() if k in ("params_changed", "within_cap", "meets_trainability",
                                                    "frozen_layers_equal", "comparisons_small", "comparisons_full",
                                                    "fidelity_small", "fidelity_full", "missing")}
    tts = out.get("test", {}).get("all", {}).get("tts_veh_h") if "test" in out else None
    print(pack, brief, "test TTS", None if tts is None else round(tts, 2), flush=True)
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for p in sys.argv[1:]:
        score(p)
