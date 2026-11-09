import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
import build_full as B
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; t = S["time_s"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
wI = impute_w2(w0); wF = B.build(w0, X)
a0 = tools.greedy(w0, X); aI = tools.greedy(wI, X); aF = tools.greedy(wF, X)
for ci, cn in enumerate(S["controller_names"]):
    m = (c == ci) & ok & (t < 3600)
    M = np.zeros((4, 4), int)
    for p, q in zip(a0[m], aF[m]): M[p, q] += 1
    print(cn, "demand-period honoured, rows start-net pick, cols full pick\n", M)
    print("   shares start", np.bincount(a0[m], minlength=4) / m.sum(), " full", np.bincount(aF[m], minlength=4) / m.sum())
# per current phase on the start_network states
m = (c == 0) & ok & (t < 3600)
for k in range(4):
    mk = m & (ph == k)
    print(f"cur phase {k}: n={mk.sum()} start picks {np.bincount(a0[mk], minlength=4)} full picks {np.bincount(aF[mk], minlength=4)}")
