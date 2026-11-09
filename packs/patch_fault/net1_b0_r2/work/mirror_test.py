"""Scenario test by 180-degree rotation (N<->S, E<->W; phases unchanged).
Take recorded states with E-side spillback (E2 full, queue in E1u), rotate them so the spillback is on the W side
(W2 full: known here). Then compare the decision of the network with the TRUE W2 values (reference, full detector)
against what a deployed network sees (W2 = 0) for different values of the unknown E2 pocket."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
w0 = tools.load("start_weights.npz")

def rot(X):
    Y = X.copy()
    # stop-line lanes N0..W2 = approach blocks of 3: N(0) E(1) S(2) W(3); rotate N<->S, E<->W
    for base in (0, 12):
        for a, b in ((0, 2), (1, 3)):
            Y[:, base + 3 * a: base + 3 * a + 3] = X[:, base + 3 * b: base + 3 * b + 3]
            Y[:, base + 3 * b: base + 3 * b + 3] = X[:, base + 3 * a: base + 3 * a + 3]
    for base in (24, 32):
        for a, b in ((0, 2), (1, 3)):
            Y[:, base + 2 * a: base + 2 * a + 2] = X[:, base + 2 * b: base + 2 * b + 2]
            Y[:, base + 2 * b: base + 2 * b + 2] = X[:, base + 2 * a: base + 2 * a + 2]
    return Y

def variants():
    wi = {k: v.copy() for k, v in w0.items()}
    wi["W1"][:, 5] += w0["W1"][:, 11]; wi["W1"][:, 17] += w0["W1"][:, 23]
    out = {"base (W2=0)": w0, "impute E2": wi}
    for beta in (2.0, 4.0):
        wb = {k: v.copy() for k, v in wi.items()}
        wb["W1"][:, 31] += beta * w0["W1"][:, 11]
        out["impute E2 + %.0f*qW1u" % beta] = wb
    return out

if __name__ == "__main__":
    m = (c > 0) & (X[:, 5] > 0.99) & (X[:, 27] >= 0.029) & (X[:, 44] > 0.5)
    print("E-spillback states (fixed+actuated, ok):", m.sum())
    R = rot(X[m])            # now W2 = 1 and W1u has the queue; E2 slot holds the (unknown) recorded W2 = 0
    for v in (0.0, 0.3348, 0.6696, 1.0):
        T = R.copy(); T[:, 5] = v; T[:, 17] = max(v, T[:, 17].max() * 0) if False else v
        ref = tools.greedy(w0, T)                 # full detector: true W2 = 1
        D = T.copy(); D[:, 11] = 0; D[:, 23] = 0   # deployed: W2 reads 0
        line = "E2=%.2f ref dist %s |" % (v, np.bincount(ref, minlength=4))
        for name, w in variants().items():
            g = tools.greedy(w, D)
            line += " %s: agree %.2f p3 %.2f |" % (name, (g == ref).mean(), (g == 3).mean())
        print(line)
    # same for all recorded states of fixed/actuated, rotated, with E2 := rotated-W2 unknown -> use E2 values as is
    m2 = (c > 0) & (X[:, 44] > 0.5)
    R = rot(X[m2]); T = R.copy(); T[:, 5] = R[:, 11]; T[:, 17] = R[:, 23]  # E2 slot = original W2 (=0): not useful
