import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UPQ = {0: [24,25,28,29], 1: [25,29], 2: [26,27,30,31], 3: [27,31]}
L = tools.forward(w, X); g = L.argmax(1); N = len(X)
q = X[:, 0:12]*capS
Qpl = np.stack([q[:, PL[p]].mean(1) for p in range(4)], 1)   # per-lane halting
Qt = np.stack([q[:, PL[p]].sum(1) for p in range(4)], 1)
U = np.stack([X[:, UPQ[p]].sum(1)*capU for p in range(4)], 1)
Qx = Qpl.copy(); Qx[np.arange(N), ph] = -9
bestpl = Qx.argmax(1)
m = ok & (g != ph)
diff = Qpl[np.arange(N), bestpl] - Qpl[np.arange(N), g]
np.set_printoptions(precision=2, suppress=True, linewidth=170)
print('switch decisions where net pick per-lane halting < best by >=1:', (m & (diff >= 1)).sum(), 'of', m.sum())
for ci in range(3):
    mm = m & (diff >= 1) & (c == ci)
    print(S['controller_names'][ci], mm.sum())
idx = np.where(m & (diff >= 1) & (c == 0))[0]
for i in idx[:25]:
    print(f'cur {ph[i]} pick {g[i]} bestpl {bestpl[i]}  Qpl {Qpl[i]}  Qtot {Qt[i]}  upq {U[i]}  dens/lane', np.round(X[i,12:24]*capS,1), ' logits', np.round(L[i],1))
