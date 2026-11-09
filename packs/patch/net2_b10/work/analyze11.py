import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
cap = 22.4/7.5; capu = 256/7.5
q = np.round(X[:, 0:12]*cap).astype(int); qu = np.round(X[:, 24:32]*capu).astype(int); du = np.round(X[:, 32:40]*capu).astype(int)
lg = tools.forward(w, X)
G = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
# For network episodes, track for each phase the time its lanes have had Q>0 continuously while not served
c = 0
waits = []
for e in range(6):
    m = (ctrl == c) & (ep == e)
    idx = np.where(m)[0]; idx = idx[np.argsort(t[idx])]
    ph = X[idx, 40:44].argmax(1)
    for p in range(4):
        start = None
        for k in range(len(idx)):
            i = idx[k]
            has = q[i, G[p]].sum() > 0 and ph[k] != p
            if has and start is None: start = k
            if (not has) and start is not None:
                waits.append((k - start) * 5); 
                if (k - start) * 5 >= 100:
                    print(f"ep {e} phase {p} waited {(k-start)*5}s from t={t[idx[start]]}")
                    for kk in range(start, k+1, 2):
                        ii = idx[kk]
                        print("   t", t[ii], "ph", ph[kk], "ok", int(X[ii,44]), "q", q[ii], "qu", qu[ii], "lg", lg[ii].round(1))
                start = None
waits = np.array(waits)
print("waits with queue (s): mean", waits.mean(), "q90", np.quantile(waits, .9), "max", waits.max())
