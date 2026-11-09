import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; A = S["act"]
print("phase one-hot sums:", np.unique(X[:, 40:44].sum(1)))
ph = X[:, 40:44].argmax(1)
for ci, cn in enumerate(S["controller_names"]):
    m = c == ci
    print(cn)
    for k in range(4):
        mk = m & (ph == k)
        mo = mk & (X[:, 44] > 0.5)
        print(f"  phase {k}: n={mk.sum():5d}  qE2 {X[mk,5].mean():.3f} dE2 {X[mk,17].mean():.3f} | qN2 {X[mk,2].mean():.3f} dN2 {X[mk,14].mean():.3f} qS2 {X[mk,8].mean():.3f} dS2 {X[mk,20].mean():.3f}"
              f"  corr(qN2,qS2) {np.corrcoef(X[mk,2],X[mk,8])[0,1]:.3f} corr(dN2,dS2) {np.corrcoef(X[mk,14],X[mk,20])[0,1]:.3f}")
# How sensitive are start-net decisions to W2 values?
ok = X[:, 44] > 0.5
base = tools.greedy(w, X)
for v in [1/3*1.005, 2/3*1.005, 1.0]:
    Xv = X.copy(); Xv[:, 11] = v; Xv[:, 23] = v
    a = tools.greedy(w, Xv)
    print(f"W2 set to {v:.3f}: decisions changed {np.mean(a != base):.3f} (honoured {np.mean(a[ok] != base[ok]):.3f}); share phase3 honoured {np.mean(a[ok]==3):.3f} vs {np.mean(base[ok]==3):.3f}")
