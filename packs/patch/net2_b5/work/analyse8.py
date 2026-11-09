import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UPS = {0: [24, 25, 28, 29], 1: [25, 29], 2: [26, 27, 30, 31], 3: [27, 31]}
g = tools.greedy(w, X); lg = tools.forward(w, X)
d = X[:, 12:24] * 3; q = X[:, :12] * 3
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
UD = np.stack([X[:, [i + 8 for i in UPS[p]]].sum(1) * 34.13 for p in range(4)], 1)
n = np.arange(len(X))
sel = np.where(ok & (g != ph) & (D[n, g] == 0) & (Q.sum(1) > 0))[0]
print('switch to empty phase while queue elsewhere:', len(sel), 'by ctrl', np.bincount(C[sel], minlength=3), 'target', np.bincount(g[sel], minlength=4), 'from', np.bincount(ph[sel], minlength=4))
for i in sel[:40]:
    print(C[i], T[i], 'cur', ph[i], 'pick', g[i], 'Q', Q[i].round(0), 'D', D[i].round(0), 'UD', UD[i].round(0), 'lg', lg[i].round(1))
