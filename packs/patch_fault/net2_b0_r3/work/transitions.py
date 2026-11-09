import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; ep = S["episode"]; t = S["time_s"]
ph = X[:, 40:44].argmax(1)
for ci in [0]:
    T = np.zeros((4, 4), int); durs = {k: [] for k in range(4)}
    for e in np.unique(ep[c == ci]):
        idx = np.where((c == ci) & (ep == e) & (t < 3600))[0]
        p = ph[idx]; g = X[idx, 45] * 60
        for a in range(1, len(idx)):
            if p[a] != p[a - 1]:
                T[p[a - 1], p[a]] += 1
                durs[p[a - 1]].append(g[a - 1] + 5)
    print("transition counts during demand (rows from, cols to)\n", T)
    for k in range(4):
        d = np.array(durs[k]); print(f"phase {k} green durations (approx s): n={len(d)} mean {d.mean():.1f} pct50/90/99/max {np.percentile(d,[50,90,99,100])}")
