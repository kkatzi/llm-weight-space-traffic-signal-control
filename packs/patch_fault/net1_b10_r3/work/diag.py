"""Diagnostics for candidate weights: idle-state decisions, agreement with v1 on recorded honoured states."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def make(c_t=0.0, k3=0.0, off=0.0):
    w0 = tools.load("start_weights.npz"); w = tools.load("start_weights.npz")
    a11, a23 = w0["W1"][:, 11], w0["W1"][:, 23]
    w["W1"][:, 5] += a11; w["W1"][:, 17] += a23          # q_W2 := q_E2, d_W2 := d_E2
    if c_t:  w["W1"][:, 45] += c_t * (a11 + a23)          # + c_t * t_green
    if k3:   w["W1"][:, 43] -= k3 * (a11 + a23)           # - k3 in phase 3
    if off:  w["b1"] += off * (a11 + a23)
    return w

def idle(w):
    out = {}
    for p in range(4):
        row = []
        for tg in [0.2, 0.4, 0.6, 0.8, 1.0]:
            x = np.zeros(46); x[40+p] = 1; x[44] = 1; x[45] = tg
            row.append(int(tools.greedy(w, x)[0]))
        out[p] = row
    return out

if __name__ == "__main__":
    S = tools.states(); X = S["obs"].astype(float); c = S["controller"]
    ok = X[:, 44] > 0.5
    w1 = make()
    a1 = tools.greedy(w1, X)
    for args in [(0,0,0),(0.5,0,0),(1,0,0),(1,0.3,0),(1.5,0.3,0),(2,0.3,0)]:
        w = make(*args)
        a = tools.greedy(w, X)
        tg = X[:, 45]
        s = f"c_t={args[0]} k3={args[1]} off={args[2]} idle={idle(w)} "
        for lo, hi in [(0.1,0.35),(0.35,0.7),(0.7,1.01)]:
            m = ok & (tg >= lo) & (tg < hi)
            s += f"| tg[{lo},{hi}) n={m.sum()} agree={np.mean(a[m]==a1[m]):.2f} p3={np.mean(a[m]==3):.2f} "
        print(s)
