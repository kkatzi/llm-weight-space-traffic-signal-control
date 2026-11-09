"""Checks on the delivered weights (forward passes only, no fitting).

Run:  python work/check.py   (from the pack folder)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tools  # noqa: E402
import rule   # noqa: E402

w = tools.load(os.path.join(HERE, "weights.npz"))
S = tools.states()
X, act, ctrl = S["obs"], S["act"], S["controller"]
ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)

print("== trainability (all recorded states)")
print({k: round(v, 4) if isinstance(v, float) else v for k, v in tools.trainability(w).items()})
print("== trainability (only states with ok=1, the decisions that change the signal)")
print({k: round(v, 4) if isinstance(v, float) else v for k, v in tools.trainability(w, X[ok]).items()})
for c, name in enumerate(S["controller_names"]):
    m = ctrl == c
    t = tools.trainability(w, X[m])
    print(f"   on {name} states: median_max_prob {t['median_max_prob']:.3f} share>0.99 "
          f"{t['share_max_prob_above_0.99']:.3f} max_abs_logit {t['max_abs_logit']:.2f}")

print("== network vs rule.py")
V = rule.scores(X)
g = tools.greedy(w, X)
srt = np.sort(V, 1)
gap = srt[:, -1] - srt[:, -2]
dis = g != V.argmax(1)
print(f"agree all {np.mean(~dis):.4f}, ok {np.mean(~dis[ok]):.4f}; disagreements {dis.sum()}, "
      f"largest rule-score gap among them {gap[dis].max() if dis.any() else 0:.4f} vehicles (ties only)")
print("tools.fidelity(rule.py):", tools.fidelity(os.path.join(HERE, "rule.py"), w))
print("count_comparisons(rule.py):", tools.count_comparisons(os.path.join(HERE, "rule.py")))

print("== network vs recorded controllers on ok=1 states")
for c, name in enumerate(S["controller_names"]):
    m = ok & (ctrl == c)
    print(f"  {name}: agree {np.mean(g[m] == act[m]):.3f}; network keeps current phase {np.mean(g[m] == ph[m]):.3f}, "
          f"controller keeps {np.mean(act[m] == ph[m]):.3f}")

print("== synthetic stress test (random states, incl. long upstream queues)")
rng = np.random.default_rng(0)
N = 20000
Z = np.zeros((N, 46))
Z[:, 0:12] = rng.integers(0, 4, (N, 12)) / 3.0
Z[:, 12:24] = np.minimum(1.0, Z[:, 0:12] + rng.integers(0, 2, (N, 12)) / 3.0)
Z[:, 24:32] = rng.choice([0, 0, 0, 1], (N, 8)) * rng.uniform(0, 0.5, (N, 8))
Z[:, 32:40] = np.minimum(1.0, Z[:, 24:32] + rng.uniform(0, 0.2, (N, 8)))
Z[np.arange(N), 40 + rng.integers(0, 4, N)] = 1.0
Z[:, 45] = rng.uniform(0, 1, N)
Z[:, 44] = (Z[:, 45] >= 10 / 60).astype(float)
Vz = rule.scores(Z)
gz = tools.greedy(w, Z)
s = np.sort(Vz, 1)
gz_gap = s[:, -1] - s[:, -2]
dz = gz != Vz.argmax(1)
print(f"agree {np.mean(~dz):.4f}; V range {Vz.min():.1f}..{Vz.max():.1f}; disagreements {dz.sum()}, "
      f"of which with rule gap > 0.05 veh: {(dz & (gz_gap > 0.05)).sum()} "
      f"(max top-score among those {Vz[dz & (gz_gap > 0.05)].max(1).min() if (dz & (gz_gap > 0.05)).any() else 0:.1f})")
t = tools.trainability(w, Z)
print("trainability on synthetic:", {k: round(v, 3) if isinstance(v, float) else v for k, v in t.items()})

print("== hand-made situations (decision, probabilities)")


def obs(cur, tg, q=None, d=None, qu=None, du=None):
    x = np.zeros(46)
    names = tools.names()
    for dct, pre in ((q, "q_"), (d, "d_"), (qu, "q_"), (du, "d_")):
        for k, v in (dct or {}).items():
            x[names.index(pre + k)] = v
    for k, v in (q or {}).items():  # density at least the queue
        x[names.index("d_" + k)] = max(x[names.index("d_" + k)], v)
    for k, v in (qu or {}).items():
        x[names.index("d_" + k)] = max(x[names.index("d_" + k)], v)
    x[40 + cur] = 1
    x[44] = 1.0 if tg >= 10 else 0.0
    x[45] = min(tg / 60, 1)
    return x


def car(n):        # n vehicles on a stop-line lane, as the observation encodes it (0.335 per car, clipped)
    return min(1.0, round(n / rule.CAP_S, 3))


def ucar(n):       # n vehicles on an upstream lane
    return min(1.0, n / rule.CAP_U)


cases = {
    "empty network, phase 2 at 12 s (expect keep 2)": obs(2, 12),
    "1 car on N left, phase 2 at 12 s (expect 1)": obs(2, 12, q={"N2": car(1)}),
    "drain: 1 car on S through only, phase 3 at 40 s (expect 0)": obs(3, 40, q={"S1": car(1)}),
    "1 car on N left, 1 car passing on E0, phase 2 at 12 s (expect keep 2, 5 s)": obs(2, 12, q={"N2": car(1)},
                                                                                     d={"E0": car(1)}),
    "EW through queue still 3+3 upstream, NS 1+1 cars (expect keep 2)": obs(2, 12, q={"N0": car(1), "S0": car(1)},
                                                                          qu={"E0u": ucar(3), "W0u": ucar(3)}),
    "full NS pockets + spillback, phase 0 at 12 s (expect 1)": obs(0, 12, q={"N2": 1, "S2": 1, "E0": car(1)},
                                                                    qu={"N1u": ucar(4), "S1u": ucar(3)}),
    "EW left 2+2, NS through 3+3, phase 2 at 12 s (expect 0)": obs(2, 12, q={"E2": car(2), "W2": car(2), "N0": 1,
                                                                            "S0": 1}),
    "heavy: NS thr 25 veh vs EW thr 20 veh, phase 1 at 12 s (expect 0)": obs(
        1, 12, q={k: 1 for k in ["N0", "N1", "S0", "S1", "E0", "E1", "W0", "W1"]},
        qu={"N0u": ucar(7), "S0u": ucar(7), "E0u": ucar(4.5), "W0u": ucar(4.5)}),
}
for k, x in cases.items():
    p = tools.probs(w, x)[0]
    print(f"  {k}: net {int(tools.greedy(w, x)[0])}, rule {rule.decide(x)}, probs {np.round(p, 3)}")
