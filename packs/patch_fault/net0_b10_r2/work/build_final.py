"""Final network = candidate D8 (see NOTES.md). Writes work/weights.npz and runs the checks."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_d as D  # noqa: E402
import impute as I  # noqa: E402
import tools  # noqa: E402

D.G = 8.0                                   # spillback weight (q_W1u -> W2)
w = D.make()
tools.save(w, os.path.join(HERE, "weights.npz"))

# ---- checks ----
w = tools.load(os.path.join(HERE, "weights.npz"))
w0 = I.W0
print("edit size:", tools.edit_size(w, w0))
changed_cols = sorted({tools.names()[j] for j in np.where(np.any(w["W1"] != w0["W1"], axis=0))[0]})
print("changed W1 columns:", changed_cols, "| other tensors changed:",
      [k for k in ["b1", "W2", "b2", "W3", "b3"] if np.any(w[k] != w0[k])])

# exact equivalence with "original network fed the estimated inputs"
S = tools.states()
X = S["obs"].astype(np.float64)
ix = I.IDX
t, ok = X[:, ix["t_green"]], X[:, ix["ok"]]
Xe = X.copy()
prior = t - D.C * ok
Xe[:, 11] = D.E * X[:, ix["q_E2"]] + D.G * X[:, ix["q_W1u"]] + D.A * prior
Xe[:, 23] = D.E * X[:, ix["d_E2"]] + D.G * X[:, ix["q_W1u"]] + D.A * prior
for ln in D.NS:
    for pre in ("q_", "d_"):
        Xe[:, ix[pre + ln]] += D.B / len(D.NS) * (t - D.CB * ok)
err = np.abs(tools.forward(w, X) - tools.forward(w0, Xe)).max()
print("max |logit difference| vs original-net-on-estimated-inputs: %.2e (float32 rounding)" % err)

hour_ok = (ok > 0.5) & (S["time_s"] < 3600)
g0, g = tools.greedy(w0, X), tools.greedy(w, X)
print("honoured demand-hour decisions changed vs start: %d of %d" % ((g0 != g)[hour_ok].sum(), hour_ok.sum()))
print("phase-3 share of honoured demand-hour decisions: start %.3f -> final %.3f"
      % ((g0[hour_ok] == 3).mean(), (g[hour_ok] == 3).mean()))
print("trainability:", tools.trainability(w))
