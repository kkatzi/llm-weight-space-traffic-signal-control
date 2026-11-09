"""build2: like build.py, plus 'offset' terms that are applied only on a subset of first-layer rows (to fit the
parameter budget). spec = {"q": {...}, "d": {...}, "partial": {"rows": K, "q": {col: coef}, "d": {col: coef}}}
Partial terms use the K rows with the largest influence of the W2 inputs (|W1[:,11]|+|W1[:,23]|, weighted by the mean
tanh slope on the recorded states and the norm of the unit's outgoing W2 column), the same rows for every partial
term so that terms meant to cancel (b1 and the phase-3 column) cancel exactly."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tools
from build import build, impute, Q, D

HERE = os.path.dirname(os.path.abspath(__file__))


def top_rows(w0, k):
    X = tools.states()["obs"].astype(np.float64)
    h1 = np.tanh(X @ w0["W1"].T + w0["b1"])
    infl = (np.abs(w0["W1"][:, Q]) + np.abs(w0["W1"][:, D])) * (1 - h1 ** 2).mean(0) * np.linalg.norm(w0["W2"], axis=0)
    return np.sort(np.argsort(-infl)[:k])


def build2(spec, w0=None):
    w0 = tools.load("start_weights.npz") if w0 is None else w0
    w = build({"q": spec.get("q", {}), "d": spec.get("d", {})}, w0)
    p = spec.get("partial")
    if p:
        R = top_rows(w0, int(p["rows"]))
        q = {int(k): float(v) for k, v in p.get("q", {}).items()}
        d = {int(k): float(v) for k, v in p.get("d", {}).items()}
        for j in sorted(set(q) | set(d)):
            add = q.get(j, 0.0) * w0["W1"][R, Q] + d.get(j, 0.0) * w0["W1"][R, D]
            if j < 0:
                w["b1"][R] += add
            else:
                w["W1"][R, j] += add
    return w


if __name__ == "__main__":
    name, spec = sys.argv[1], json.loads(sys.argv[2])
    w0 = tools.load("start_weights.npz")
    w = build2(spec, w0)
    es = tools.edit_size(w, w0)
    X = tools.states()["obs"].astype(np.float64)
    full = {"q": dict(spec.get("q", {})), "d": dict(spec.get("d", {}))}
    for k in ("q", "d"):
        for j, v in spec.get("partial", {}).get(k, {}).items():
            full[k][j] = full[k].get(j, 0) + v
    a_exact = tools.greedy(w0, impute(full, X)); a = tools.greedy(w, X)
    ok = X[:, 44] > 0.5
    print("edit size:", es, " agreement with exact (unbudgeted) imputation on honoured states:",
          round(float((a == a_exact)[ok].mean()), 4))
    assert es["changed"] <= 371
    path = os.path.join(HERE, name + ".npz")
    tools.save(w, path)
    print("after float32 save:", tools.edit_size(tools.load(path), w0))
    with open(os.path.join(HERE, name + ".json"), "w") as f:
        json.dump(spec, f)
    print("saved", path)
