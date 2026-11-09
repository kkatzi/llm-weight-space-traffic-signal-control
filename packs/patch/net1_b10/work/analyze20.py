import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UPQ = {0: [24,25,28,29], 1: [25,29], 2: [26,27,30,31], 3: [27,31]}
g = tools.greedy(w, X); N = len(X)
q = X[:, 0:12]*capS
Qpl = np.stack([q[:, PL[p]].mean(1) for p in range(4)], 1)
U = np.stack([X[:, UPQ[p]].sum(1)*capU for p in range(4)], 1)
np.set_printoptions(precision=2, suppress=True, linewidth=170)
sat = (Qpl >= 2.5)
sat[np.arange(N), ph] = False
m = ok & (sat.sum(1) >= 2)
print('ok states with >=2 candidate phases saturated at stop line:', m.sum(), ' by controller', np.bincount(c[m], minlength=3))
# among saturated candidates, which has largest upstream queue; does net pick it?
agree = 0; picks_sat = 0; tot = 0; rows = []
for i in np.where(m)[0]:
    cands = np.where(sat[i])[0]
    best = cands[np.argmax(U[i, cands])]
    tot += 1
    picks_sat += g[i] in cands
    agree += (g[i] == best)
    if U[i, cands].max() - np.sort(U[i, cands])[-2] >= 3 and g[i] != best:
        rows.append(i)
print('net picks a saturated candidate', picks_sat, '/', tot, ' picks the one with largest upstream queue', agree)
print('clear disagreements (upstream gap >= 3 veh):', len(rows))
for i in rows[:20]:
    print(f'  ctl {c[i]} cur {ph[i]} pick {g[i]} Qpl {Qpl[i]} upQ {U[i]}')
