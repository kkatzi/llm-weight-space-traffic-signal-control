"""Probes of the delivered network on recorded and synthetic L2W inputs (forward passes only)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(HERE)
sys.path.insert(0, PACK)
sys.path.insert(0, HERE)
import tools  # noqa: E402
from build import N46  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=160)
N54 = tools.names()
S = tools.states()
X = S["obs"].astype(np.float64)
Y = tools.states("states_L2.npz")["obs"].astype(np.float64)
w = tools.load(os.path.join(HERE, "weights.npz"))
src = tools.load(os.path.join(PACK, "source_weights_L2.npz"))

LANES54 = {0: ["N0", "N1", "S0", "S1"], 1: ["N2", "S2"], 2: ["E0", "E1", "E2", "W0", "W1", "W2"], 3: ["E3", "W3"]}
LANES46 = {0: ["N0", "N1", "S0", "S1"], 1: ["N2", "S2"], 2: ["E0", "E1", "W0", "W1"], 3: ["E2", "W2"]}


def pressure(Z, names, lanes):
    """Mean queue+density per served stop-line lane, per phase."""
    P = np.zeros((len(Z), 4))
    for p, ls in lanes.items():
        P[:, p] = np.mean([Z[:, names.index(f"q_{l}")] + Z[:, names.index(f"d_{l}")] for l in ls], axis=0)
    return P


def consistency(Z, names, lanes, a, okcol):
    P = pressure(Z, names, lanes)
    ph = Z[:, -6:-2].argmax(1)
    m = Z[:, okcol] > 0.5
    # among honoured decisions: pick vs highest-pressure phase other than the current one
    Pm = P.copy()
    Pm[np.arange(len(Z)), ph] = -1
    best_other = Pm.argmax(1)
    rank = (-Pm).argsort(1).argsort(1)[np.arange(len(Z)), a]  # 0 = best other phase
    sw = m & (a != ph)
    return {"n_ok": int(m.sum()), "switch_share": round(float(sw.sum() / m.sum()), 3),
            "pick=best_other": round(float((a[sw] == best_other[sw]).mean()), 3),
            "pick_rank_dist": np.bincount(rank[sw], minlength=4) / sw.sum()}


print("pressure consistency, L2 net on L2:", consistency(Y, N46, LANES46, tools.greedy(src, Y), 44))
print("pressure consistency, new net on L2W:", consistency(X, N54, LANES54, tools.greedy(w, X), 52))

# spillback probes: start from recorded ok=1 states, push one upstream lane to a level, see the pick
ok = X[:, 52] > 0.5
base = X[ok]
print("\nupstream spillback probes (share of picks per phase, on", len(base), "ok-states):")
for lane, partner in [("E2u", "W2u"), ("E0u", "E1u"), ("N1u", None), ("N0u", None)]:
    for lev in (0.0, 0.2, 0.5, 1.0):
        Z = base.copy()
        for ln in [lane] + ([partner] if partner else []):
            Z[:, N54.index(f"q_{ln}")] = lev
            Z[:, N54.index(f"d_{ln}")] = lev
        a = tools.greedy(w, Z)
        print(f"  {lane}{'+' + partner if partner else ''} q=d={lev:.1f}: picks {np.bincount(a, minlength=4) / len(a)}")

# synthetic single-movement demand: one phase's lanes full, others empty, every current phase, ok=1
print("\nsynthetic: only one phase's stop-line lanes full (q=d=1), ok=1, t_green=0.2")
for target, ls in LANES54.items():
    row = []
    for cur in range(4):
        z = np.zeros(54)
        for l in ls:
            z[N54.index(f"q_{l}")] = 1
            z[N54.index(f"d_{l}")] = 1
        z[48 + cur] = 1
        z[52] = 1
        z[53] = 0.2
        row.append(int(tools.greedy(w, z)[0]))
    print(f"  demand only on phase {target}: pick for current phase 0..3 = {row}")

# per-scenario-like split: recorded episodes by load (mean stop-line occupancy) and the picks
print("\nhigh-load L2W states (top 20% mean density): act dist",
      np.bincount(tools.greedy(w, X[X[:, 14:28].mean(1) > np.quantile(X[:, 14:28].mean(1), .8)]), minlength=4) /
      int((X[:, 14:28].mean(1) > np.quantile(X[:, 14:28].mean(1), .8)).sum()))
