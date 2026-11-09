import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']; t = S['time_s']; ep = S['episode']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
PU = {0: [24,25,28,29], 1: [25,29], 2: [26,27,30,31], 3: [27,31]}
g = tools.greedy(w, X); N = len(X)
q = X[:, 0:12]*capS; d = X[:, 12:24]*capS
Qcur = np.array([q[i, PL[ph[i]]].sum() for i in range(N)])
Dcur = np.array([d[i, PL[ph[i]]].sum() for i in range(N)])
Ucur = np.array([X[i, PU[ph[i]]].sum()*capU for i in range(N)])
UDcur = np.array([X[i, [k+8 for k in PU[ph[i]]]].sum()*capU for i in range(N)])
for ci in range(3):
    m = ok & (c == ci)
    print(S['controller_names'][ci])
    for lo, hi in [(0, 0.5), (0.5, 1.5), (1.5, 3.5), (3.5, 20)]:
        mm = m & (Qcur >= lo) & (Qcur < hi)
        print(f'  current-phase halting in [{lo},{hi}): n={mm.sum()} net hold={(g[mm]==ph[mm]).mean() if mm.sum() else 0:.2f}')
    for lo, hi in [(0, 1), (1, 3), (3, 6), (6, 50)]:
        mm = m & (Dcur >= lo) & (Dcur < hi)
        print(f'  current-phase stopline vehicles in [{lo},{hi}): n={mm.sum()} net hold={(g[mm]==ph[mm]).mean() if mm.sum() else 0:.2f}')
# Next-state check: in start-net episodes, after a switch at t_green 12, how many halting at the left-behind phase 5 s later?
