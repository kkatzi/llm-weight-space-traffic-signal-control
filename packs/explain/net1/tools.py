"""tools.py: helpers for working on the network's weights by hand.

    import tools
    w = tools.load("start_weights.npz")       # dict W1 b1 W2 b2 W3 b3 (float64)
    S = tools.states()                        # recorded observations (see TASK.md)
    a = tools.greedy(w, S["obs"])             # [N] chosen phase per state
    tools.block(S["obs"][0])                  # one observation's features by name
    tools.names()                             # the observation entries' names, in order
    tools.save(w, "work/weights.npz")         # checks shapes and finiteness
    tools.edit_size(w, tools.load("start_weights.npz"))
    tools.trainability(w)                     # how easy the network is for PPO to continue training
    tools.fidelity("work/rules_small.py")     # agreement of a rules file with the network
    tools.count_comparisons("work/rules_small.py")

Forward pass (identical to evaluation):
    h1 = tanh(W1 @ x + b1);  h2 = tanh(W2 @ h1 + b2);  logits = W3 @ h2 + b3;  phase = argmax(logits)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["W1", "b1", "W2", "b2", "W3", "b3"]
with open(os.path.join(HERE, "spec.json")) as _f:
    SPEC = json.load(_f)


def shapes(spec=SPEC):
    o, h = spec["obs_dim"], spec["hidden"]
    return {"W1": (h, o), "b1": (h,), "W2": (h, h), "b2": (h,), "W3": (4, h), "b3": (4,)}


def load(path):
    path = path if os.path.isabs(path) or os.path.exists(path) else os.path.join(HERE, path)
    z = np.load(path)
    return {k: z[k].astype(np.float64) for k in KEYS}


def save(w, path):
    want = shapes()
    for k in KEYS:
        a = np.asarray(w[k], dtype=np.float64)
        if a.shape != want[k]:
            raise ValueError(f"{k}: shape {a.shape}, expected {want[k]}")
        if not np.isfinite(a).all():
            raise ValueError(f"{k}: non-finite values")
    np.savez(path, **{k: np.asarray(w[k], dtype=np.float32) for k in KEYS})
    return path


def forward(w, X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    h = np.tanh(X @ w["W1"].T + w["b1"])
    h = np.tanh(h @ w["W2"].T + w["b2"])
    return h @ w["W3"].T + w["b3"]


def hidden(w, X):
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    h1 = np.tanh(X @ w["W1"].T + w["b1"])
    h2 = np.tanh(h1 @ w["W2"].T + w["b2"])
    return h1, h2


def greedy(w, X):
    return forward(w, X).argmax(axis=1)


def probs(w, X):
    lg = forward(w, X)
    e = np.exp(lg - lg.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def states(name="states.npz"):
    z = np.load(os.path.join(HERE, name), allow_pickle=False)
    return {k: z[k] for k in z.files}


def names():
    return list(SPEC["obs_names"])


def block(x):
    """One observation's features by name."""
    x = np.asarray(x, dtype=np.float64)
    return {n: round(float(v), 3) for n, v in zip(SPEC["obs_names"], x)}


def edit_size(w_new, w_ref, tol=0.0):
    changed = sum(int((np.abs(np.asarray(w_new[k]) - np.asarray(w_ref[k])) > tol).sum()) for k in KEYS)
    total = sum(np.asarray(w_ref[k]).size for k in KEYS)
    return {"changed": changed, "total": total, "fraction": changed / total}


def trainability(w, X=None):
    """Numbers that tell whether PPO can continue training this network smoothly, on the recorded states (default)
    or on X.

    max_prob     probability of the most likely phase, per state. PPO samples from these probabilities while
                 training. Near 1.0 everywhere means no exploration, and tiny weight changes flip decisions, which
                 makes the first PPO updates destructive.
    logit_gap    top-1 minus top-2 logit
    sat_l1/l2    share of (state, unit) pairs with |pre-activation| > 3 (tanh saturated, almost no gradient)
    dead_l1/l2   units whose activation never varies (std < 0.01) over the states
    """
    X = states()["obs"] if X is None else np.atleast_2d(np.asarray(X, dtype=np.float64))
    a1 = X @ w["W1"].T + w["b1"]
    h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]
    h2 = np.tanh(a2)
    lg = h2 @ w["W3"].T + w["b3"]
    p = np.exp(lg - lg.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    mp = p.max(axis=1)
    srt = np.sort(lg, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    return {"median_max_prob": float(np.median(mp)), "share_max_prob_above_0.99": float((mp > 0.99).mean()),
            "median_logit_gap": float(np.median(gap)), "max_abs_logit": float(np.abs(lg).max()),
            "sat_l1": float((np.abs(a1) > 3).mean()), "sat_l2": float((np.abs(a2) > 3).mean()),
            "dead_l1": int((h1.std(axis=0) < 0.01).sum()), "dead_l2": int((h2.std(axis=0) < 0.01).sum())}


def _load_rules(path):
    import importlib.util
    path = path if os.path.isabs(path) else os.path.join(HERE, path)
    spec = importlib.util.spec_from_file_location("rules_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.decide


def fidelity(rules_path, w=None, X=None):
    """Share of states where rules_path's decide(x) picks the same phase as the network w (default
    start_weights.npz), on the recorded states (default) or X. 'honoured' counts only states where the minimum
    green has elapsed (x[ok] == 1), the only decisions that change the signal; 'all' counts every state."""
    decide = _load_rules(rules_path)
    w = load("start_weights.npz") if w is None else w
    X = states()["obs"] if X is None else np.atleast_2d(np.asarray(X, dtype=np.float64))
    net = greedy(w, X)
    rul = np.array([int(decide(x.astype(np.float64))) for x in X])
    ok = X[:, SPEC["ok_index"]] > 0.5
    return {"honoured": float((net[ok] == rul[ok]).mean()), "all": float((net == rul).mean()),
            "n_honoured": int(ok.sum())}


def count_comparisons(rules_path):
    """Number of comparison expressions (a < b, a >= b, ...) in the file: the size measure."""
    import ast
    path = rules_path if os.path.isabs(rules_path) else os.path.join(HERE, rules_path)
    tree = ast.parse(open(path).read())
    return sum(len(n.ops) for n in ast.walk(tree) if isinstance(n, ast.Compare))
