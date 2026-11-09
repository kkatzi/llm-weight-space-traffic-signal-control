import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap = 22.4/7.5; capu = 256/7.5
q = X[:, 0:12]*cap
G = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
for c in [0, 2]:
    reds = {p: [] for p in range(4)}
    for e in range(6):
        m = (ctrl == c) & (ep == e)
        idx = np.where(m)[0]; idx = idx[np.argsort(t[idx])]
        ph = X[idx, 40:44].argmax(1); tt = t[idx]
        last_end = {p: None for p in range(4)}
        for k in range(len(idx)):
            p = ph[k]
            if k > 0 and ph[k-1] != p:
                # phase p starts at tt[k]; red since last end
                if last_end[p] is not None:
                    reds[p].append(tt[k] - last_end[p])
            last_end[p] = tt[k]
    print("ctrl", S["controller_names"][c])
    for p in range(4):
        r = np.array(reds[p])
        print(f"  phase {p}: n={len(r)} red mean {r.mean():.1f} q90 {np.quantile(r,0.9):.0f} q99 {np.quantile(r,0.99):.0f} max {r.max():.0f}")
