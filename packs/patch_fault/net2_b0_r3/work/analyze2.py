import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; A = S["act"]
nm = tools.names()
print("unique d_E2 values", np.unique(np.round(X[:, 17], 3)))
print("unique q_E2 values", np.unique(np.round(X[:, 5], 3)))
for ci, cn in enumerate(S["controller_names"]):
    m = c == ci
    Xm = X[m]
    def cr(a, b): return np.corrcoef(Xm[:, a], Xm[:, b])[0, 1]
    print(cn, "corr qN2-qS2 %.3f dN2-dS2 %.3f | qE2-qN2 %.3f dE2-dN2 %.3f | qE2-dE2 %.3f qN2-dN2 %.3f" % (
        cr(2, 8), cr(14, 20), cr(5, 2), cr(17, 14), cr(5, 17), cr(2, 14)))
    print("   means qN2 %.3f qS2 %.3f qE2 %.3f | dN2 %.3f dS2 %.3f dE2 %.3f | dW1u %.3f dE1u %.3f qW1u %.3f qE1u %.3f" % tuple(
        Xm[:, [2, 8, 5, 14, 20, 17, 39, 35, 31, 27]].mean(0)))
    print("   action shares", np.bincount(A[m], minlength=4) / m.sum())
    print("   phase shares", Xm[:, 40:44].mean(0))
