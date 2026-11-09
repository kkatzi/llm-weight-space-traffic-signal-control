import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
q = X[:, 0:12]*capS; d = X[:, 12:24]*capS; qu = X[:, 24:32]*capU; du = X[:, 32:40]*capU
# per-phase stopline queue
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
PU = {0: [0,1,4,5], 1: [1,5], 2: [2,3,6,7], 3: [3,7]}  # upstream lanes feeding (shared for 1/3)
def phase_q(X):
    q = X[:, 0:12]*capS; qu = X[:, 24:32]*capU
    out = np.zeros((len(X), 4))
    for p in range(4): out[:, p] = q[:, PL[p]].sum(1)
    return out
def phase_d(X):
    d = X[:, 12:24]*capS
    out = np.zeros((len(X), 4))
    for p in range(4): out[:, p] = d[:, PL[p]].sum(1)
    return out
g = tools.greedy(w, X)
Q = phase_q(X); D = phase_d(X)
m = ok
print('n ok states', m.sum())
# picks of net vs argmax queue excluding current
Qx = Q.copy(); Qx[np.arange(len(X)), ph] = -1
best = Qx.argmax(1)
for ci in range(3):
    mm = m & (c == ci)
    print(S['controller_names'][ci], 'net pick==max-queue other phase', (g[mm]==best[mm]).mean(), ' net picks phase with zero stopline queue & zero density:', ((D[mm, g[mm]] < 0.01) & (g[mm]!=ph[mm])).mean())
# where net picks a phase with zero density
mm = m & (c==0)
z = mm & (D[np.arange(len(X)), g] < 0.01) & (g != ph)
print('start-net ok states picking empty phase:', z.sum(), 'of', mm.sum())
# time profile: total vehicles vs time for each controller
for ci in range(3):
    mm = c == ci
    veh = (X[mm][:, 12:24]*capS).sum(1) + (X[mm][:, 32:40]*capU).sum(1)
    tt = t[mm]
    bins = np.arange(0, 3800, 600)
    prof = [veh[(tt>=b)&(tt<b+600)].mean() for b in bins]
    print(S['controller_names'][ci], 'mean veh observed per 10 min:', np.round(prof,1), ' total veh-s/3600 per ep:', round(veh.sum()*5/3600/6,2))
