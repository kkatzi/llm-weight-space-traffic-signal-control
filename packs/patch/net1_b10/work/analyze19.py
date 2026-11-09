import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']; ep = S['episode']; t = S['time_s']; A = S['act']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
q = X[:, 0:12]*capS
Q = np.stack([q[:, PL[p]].sum(1) for p in range(4)], 1)
N = len(X)
Qother = Q.sum(1) - Q[np.arange(N), ph]
np.set_printoptions(precision=2, suppress=True, linewidth=170)
m0 = (c == 0)
hold = ok & (A == ph) & m0
print('start-net honoured holds:', hold.sum(), ' with other halting>0:', (hold & (Qother > 0.5)).sum())
print('  veh-s of other-phase halting during these holds (x5 s):', (Qother[hold & (Qother > 0.5)]*5).sum(), ' => veh-h per episode', (Qother[hold & (Qother>0.5)]*5).sum()/3600/6)
# runs of holds: longest
for e in range(6):
    idx = np.where(m0 & (ep == e))[0]
    hh = hold[idx] & (Qother[idx] > 0.5)
    tt = t[idx]
    # episodes' last 300 s
    print(f' ep {e}: end time {tt.max()}, holds-with-waiting {hh.sum()}, times', tt[hh][:20])
# drain-time decisions: t > 3600
d = m0 & (t > 3600)
print('drain decisions (t>3600):', d.sum())
for i in np.where(d)[0][:40]:
    print(t[i], 'ph', ph[i], 'ok', int(ok[i]), 'act', A[i], 'Q', Q[i], 'dens', np.round(X[i, 12:24]*capS, 1), 'updens', np.round(X[i, 32:40]*capU, 1))
