"""Offline checks of a weights file against v1 (forward passes only)."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "work", "v2.npz")
w = tools.load(path); w1 = tools.load(os.path.join(HERE, "work", "v1.npz")); w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
print("edit size", tools.edit_size(w, w0))
St = tools.states(); X = St["obs"].astype(float); c = St["controller"]; ep = St["episode"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
J_LONG, J_SPILL, K_RELAY = 0, 31, 25
h1, h2 = tools.hidden(w, X)
g1 = tools.greedy(w1, X); g2 = tools.greedy(w, X)
fire = h2[:, K_RELAY] > 0
for nm, m in [("start ep0-1", (c == 0) & (ep < 2)), ("start ep2-5 (gridlock)", (c == 0) & (ep >= 2)), ("fixed", c == 1), ("actuated", c == 2)]:
    nf = m & ~fire
    print(f"{nm:24s} relay fires {fire[m].mean():.4f} (long {(h1[m, J_LONG] > 0).mean():.4f}, spill {(h1[m, J_SPILL] > 0).mean():.4f});"
          f" honoured non-firing states changed vs v1: {(g1[nf & ok] != g2[nf & ok]).mean():.4f};  firing states -> phase3: {(g2[m & fire] == 3).mean() if (m & fire).any() else float('nan'):.3f}")
# relay must be cleanly saturated
print("relay |h2| min over states", np.abs(h2[:, K_RELAY]).min(), " h1 trigger |.| min", np.abs(h1[:, [J_LONG, J_SPILL]]).min(0))
# idle junction: every working detector empty
print("idle junction, decision by (phase, t_green):")
for p in range(4):
    row = []
    for g in [2, 12, 22, 32, 42, 52, 57, 62]:
        x = np.zeros(46); x[40 + p] = 1; x[45] = min(g / 60, 1); x[44] = float(g >= 10)
        row.append(f"{g}s->{tools.greedy(w, x)[0]}")
    print("  phase", p, " ".join(row))
# idle junction except a W1u queue of 1-2 cars and W1 empty, phase 2 green (stranded left-turner case)
for n in [1, 2, 5]:
    x = np.zeros(46); x[42] = 1; x[45] = 22 / 60; x[44] = 1; x[31] = n / (256 / 7.5); x[39] = n / (256 / 7.5)
    print(f"  phase2 22s, {n} halting on W1u, W1 empty -> v1 {tools.greedy(w1, x)[0]}, new {tools.greedy(w, x)[0]}")
# mirrored E-side spillback states: E-side signature (phase 2, q_E1u>0, d_E1=0) swapped onto the W side
names = tools.names()
idx = {n: i for i, n in enumerate(names)}
def mirror(X):
    Y = X.copy()
    pairs = [("N", "S"), ("E", "W")]
    for a_, b_ in pairs:
        for suf in ["0", "1", "2"]:
            for pre in ["q_", "d_"]:
                ia, ib = idx[pre + a_ + suf], idx[pre + b_ + suf]
                Y[:, ia], Y[:, ib] = X[:, ib], X[:, ia]
        for suf in ["0u", "1u"]:
            for pre in ["q_", "d_"]:
                ia, ib = idx[pre + a_ + suf], idx[pre + b_ + suf]
                Y[:, ia], Y[:, ib] = X[:, ib], X[:, ia]
    Y[:, 11] = 0; Y[:, 23] = 0       # the W2 detector is dead
    return Y
good = (c > 0) | (ep < 2)
sig = good & (ph == 2) & (X[:, 27] > 0) & (X[:, 16] == 0) & ok
print("E-side spillback states:", sig.sum(), " q_E2 dist", np.bincount(np.round(X[sig, 5] * 3).astype(int), minlength=4) / sig.sum())
Y = mirror(X[sig])
print("  unmirrored (E side, E2 visible): v1 picks", np.bincount(g1[sig], minlength=4))
print("  mirrored onto W side (E2 now = old W2 obs = 0): v1 picks", np.bincount(tools.greedy(w1, Y), minlength=4),
      " new picks", np.bincount(tools.greedy(w, Y), minlength=4))
print(tools.trainability(w))
