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
q = np.round(X[:, :12] * 3); d = np.round(X[:, 12:24] * 3)
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
D = np.stack([d[:, SERVE[p]].sum(1) for p in range(4)], 1)
n = np.arange(len(X))
Qo = Q.copy(); Qo[n, ph] = -1
mx = Qo.max(1)
sel = np.where(ok & (Q[n, g] <= 1) & (mx >= 6))[0]
print('picked <=1 halting while another phase >=6:', len(sel), 'by ctrl', np.bincount(C[sel], minlength=3), 'hold', (g[sel] == ph[sel]).sum())
for i in sel[:30]:
    print(C[i], T[i], 'cur', ph[i], 'tg', round(X[i, 45]*60), 'pick', g[i], 'Q', Q[i], 'D', D[i], 'uq', (X[i, 24:32]*34.13).round(0), 'lg', lg[i].round(1))
