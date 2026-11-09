import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); c = S['controller']; t = S['time_s']
capS = 22.4/7.5; capU = 256/7.5
q = X[:, 0:12]*capS; qu = X[:, 24:32]*capU
np.set_printoptions(precision=2, suppress=True, linewidth=160)
for ci in range(3):
    m = c == ci
    print(S['controller_names'][ci], 'mean halting per stop-line lane', q[m].mean(0), ' upstream', qu[m].mean(0))
    print('   by phase group: NS thr', q[m][:, [0,1,6,7]].sum(1).mean(), ' NS left', q[m][:, [2,8]].sum(1).mean(), ' EW thr', q[m][:, [3,4,9,10]].sum(1).mean(), ' EW left', q[m][:, [5,11]].sum(1).mean(), ' upstream', qu[m].sum(1).mean())
    # fraction of time stop-line left lane is full (3)
    print('   fraction time lane at cap (q=1):', (X[m][:, 0:12] > 0.99).mean(0))
