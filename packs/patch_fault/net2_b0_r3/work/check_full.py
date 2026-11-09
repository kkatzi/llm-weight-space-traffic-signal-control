import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
import build_full as B
from chain import deadlock_report
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; t = S["time_s"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
wI = impute_w2(w0)
wF = B.build(w0, X)
aI = tools.greedy(wI, X); aF = tools.greedy(wF, X); a0 = tools.greedy(w0, X)
h1, h2 = tools.hidden(wF, X)
condA = (ph == 2) & (X[:, 45] >= B.THETA_A)
condB = (ph == 3) & (X[:, 45] >= B.THETA_B) & (X[:, 5] == 0) & (X[:, 17] == 0)
print("u_A h1 range normal", h1[~condA, B.U_A].min(), h1[~condA, B.U_A].max(), " cond", h1[condA, B.U_A].min() if condA.any() else None, h1[condA, B.U_A].max() if condA.any() else None)
print("u_B h1 range normal", h1[~condB, B.U_B].min(), h1[~condB, B.U_B].max(), " cond", h1[condB, B.U_B].min(), h1[condB, B.U_B].max())
print("j_A h2 normal", h2[~condA, B.J_A].min(), h2[~condA, B.J_A].max(), " j_B h2 normal", h2[~condB, B.J_B].min(), h2[~condB, B.J_B].max())
print("cond A states:", condA.sum(), " during demand:", (condA & (t < 3600)).sum())
print("cond B states:", condB.sum(), " during demand:", (condB & (t < 3600)).sum(), " honoured:", (condB & ok).sum())
m = condB & ok & (t < 3600)
print("  under cond B (demand, honoured): imputed-net picks", np.bincount(aI[m], minlength=4), " start net picks", np.bincount(a0[m], minlength=4), " full picks", np.bincount(aF[m], minlength=4))
other = ~(condA | condB)
print("decision changes full vs imputation-only outside conditions: all %.4f honoured %.4f" % (np.mean(aI[other] != aF[other]), np.mean(aI[other & ok] != aF[other & ok])))
print("decision changes imputation vs start: all %.4f honoured %.4f" % (np.mean(aI != a0), np.mean(aI[ok] != a0[ok])))
for ci, cn in enumerate(S["controller_names"]):
    mm = (c == ci) & ok
    print(f"  {cn:13s} honoured: phase shares start {np.bincount(a0[mm], minlength=4)/mm.sum()} -> full {np.bincount(aF[mm], minlength=4)/mm.sum()}")
deadlock_report(wF, "full build")
