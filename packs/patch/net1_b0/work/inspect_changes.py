import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from collections import Counter
from build import build, ideal_logits, H_coef, LAM
np.set_printoptions(precision=1, suppress=True, linewidth=220)
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75; M = float(sys.argv[2]) if len(sys.argv) > 2 else 2.5
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5; ctl = S["controller"]; t = S["time_s"]; ep = S["episode"]
w = build(w0, alpha=alpha, M=M, X_mean=X[ok], k_units=(31, 27, 59), j_units=(25, 35, 41))
lg0 = tools.forward(w0, X); lg1 = tools.forward(w, X); g0 = lg0.argmax(1); g1 = lg1.argmax(1)
H = X @ H_coef().T
ph = X[:, 40:44].argmax(1); I = np.arange(len(X))
m = ok & (ctl == 0) & (g0 != g1)
print("changed ctl0:", m.sum(), "of", (ok & (ctl == 0)).sum())
print("orig gap of changed: quantiles", np.quantile(lg0[I, g0][m] - lg0[I, g1][m], [.25, .5, .75, .9, 1]))
print("H(new) - H(old) quantiles:", np.quantile(H[I, g1][m] - H[I, g0][m], [0, .1, .25, .5, .75, .9, 1]))
print("pairs (cur, old, new):", Counter(zip(ph[m], g0[m], g1[m])).most_common(15))
print("holds orig/new (ctl0 ok):", (g0[ok & (ctl==0)] == ph[ok & (ctl==0)]).sum(), (g1[ok & (ctl==0)] == ph[ok & (ctl==0)]).sum())
print("pick distribution orig:", np.bincount(g0[ok & (ctl==0)], minlength=4), " new:", np.bincount(g1[ok & (ctl==0)], minlength=4))
for i in np.where(m)[0][:25]:
    print(f"ep{ep[i]} t={t[i]:4d} cur={ph[i]} old={g0[i]} new={g1[i]} H={H[i]} lg0={lg0[i].round(1)}")
