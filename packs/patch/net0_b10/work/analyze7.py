import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']
ix = {n:i for i,n in enumerate(tools.names())}
SL = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
UP = {0: ['N0u', 'N1u', 'S0u', 'S1u'], 1: ['N1u', 'S1u'], 2: ['E0u', 'E1u', 'W0u', 'W1u'], 3: ['E1u', 'W1u']}
cs, cu = 22.4/7.5, 256/7.5
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5
m = (ctrl==0)&ok
qcur = np.array([sum(X[i, ix['q_'+l]] for l in SL[ph[i]])*cs for i in range(len(X))])
qup = np.array([sum(X[i, ix['q_'+u]] for u in UP[ph[i]])*cu for i in range(len(X))])
dcur = np.array([sum(X[i, ix['d_'+l]] for l in SL[ph[i]])*cs for i in range(len(X))])
print('net ok-states: halted vehicles on current phase stop-line lanes (veh):', np.bincount(np.round(qcur[m]).astype(int)))
print('   vehicles (any speed) on current stop-line lanes:', np.bincount(np.round(dcur[m]).astype(int)))
print('   halted on current upstream lanes:', np.bincount(np.round(qup[m]).astype(int)))
# by phase
for p in range(4):
    mm = m & (ph==p)
    print('phase', p, 'n', mm.sum(), 'mean halted cur SL %.2f, frac >=2 halted %.2f' % (qcur[mm].mean(), (qcur[mm]>=1.5).mean()))
