"""Synthetic-state sanity checks and perturbation robustness of work/weights.npz (forward passes only)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np, tools, rule
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load(os.path.join(HERE, "weights.npz"))
ix = {n: i for i, n in enumerate(tools.names())}
CU = 256 / 7.5

def state(phase, ok, tg, **kv):
    x = np.zeros(46); x[40 + phase] = 1; x[44] = ok; x[45] = min(tg / 60, 1)
    for k, v in kv.items(): x[ix[k]] = v
    for l in ["N0", "N1", "N2", "E0", "E1", "E2", "S0", "S1", "S2", "W0", "W1", "W2"]:   # density >= queue
        x[ix["d_" + l]] = max(x[ix["d_" + l]], x[ix["q_" + l]])
    for l in ["N0u", "N1u", "E0u", "E1u", "S0u", "S1u", "W0u", "W1u"]:
        x[ix["d_" + l]] = max(x[ix["d_" + l]], x[ix["q_" + l]])
    return x

cases = [
    ("empty network, EW_through green, ok", state(2, 1, 12), 2),
    ("EW_through green ok, NS through queues full", state(2, 1, 12, q_N0=1, q_N1=1, q_S0=1, q_S1=1), 0),
    ("NS_through green ok, EW left pockets full + E1u spillback 5", state(0, 1, 12, q_E2=1, q_W2=1, q_E1u=5 / CU, q_E0=.33, q_W0=.33), 3),
    ("NS_through green ok, EW through 2/lane, EW left 1 each", state(0, 1, 12, q_E0=.67, q_E1=.67, q_W0=.67, q_W1=.67, q_E2=.33, q_W2=.33), 2),
    ("EW_through green ok, own queue still discharging (moving 3/lane + 8 upstream moving), others 1-2", state(2, 1, 12, d_E0=1, d_E1=1, d_W0=1, d_W1=1, d_E0u=8 / CU, d_W0u=8 / CU, q_N2=.33, q_S0=.33), 2),
    ("EW_through green ok, long own upstream queue (7 halting on E0u), NS through 1/lane", state(2, 1, 12, q_E0u=7 / CU, d_E0=1, q_N0=.33, q_N1=.33, q_S0=.33, q_S1=.33), 2),
    ("EW_left green ok=0 (t=2), NS through queues full", state(3, 0, 2, q_N0=1, q_N1=1, q_S0=1, q_S1=1), None),
    ("EW_left green ok=0 (t=7), light demand elsewhere", state(3, 0, 7, q_N0=.33, q_E0=.33), 3),
    ("NS_left green ok, nothing waiting, EW through platoon approaching (6 moving upstream)", state(1, 1, 12, d_E0u=3 / CU, d_E1u=3 / CU, d_W0u=3 / CU), 2),
]
print("synthetic cases: decision, probabilities, expected")
for name, x, exp in cases:
    p = tools.probs(w, x)[0]; a = int(p.argmax())
    flag = "" if exp is None else ("OK" if a == exp else "UNEXPECTED")
    print(f"  {name:95s} -> {a}  p={p}  exp={exp} {flag}")

S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5
net = tools.greedy(w, X); D = rule.scores(X); Ds = np.sort(D, 1); gap = Ds[:, -1] - Ds[:, -2]
rng = np.random.default_rng(1)
print("perturbation robustness (Gaussian noise on every weight, 20 draws each), ok states:")
for s in [0.003, 0.01, 0.03]:
    fl, big = [], []
    for _ in range(20):
        w2 = {k: v + rng.normal(0, s, v.shape) for k, v in w.items()}
        f = tools.greedy(w2, X[ok]) != net[ok]
        fl.append(f.mean()); big.append((f & (gap[ok] > 2)).mean())
    print(f"  sigma {s}: flip rate {np.mean(fl):.4f} (flips where rule gap > 2 vehicles: {np.mean(big):.4f})")
