import numpy as np, sys, copy
sys.path.insert(0, '.')
import tools
exec(open('work/analyze17.py').read().split('for alpha in')[0])
capS = 22.4/7.5
q = X[:, 0:12]*capS
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
Q = np.stack([q[:, PL[p]].sum(1) for p in range(4)], 1)
np.set_printoptions(precision=2, suppress=True, linewidth=170)
for alpha in [0.75, 1.3]:
    ws = scaled(alpha, LEFT_Q + LEFT_D, topk=40)
    gs = tools.greedy(ws, X)
    nh = m & (gs != g) & (gs == ph)
    print(f'alpha {alpha}: new holds {nh.sum()} by current phase', np.bincount(ph[nh], minlength=4))
    print('   mean halting per phase in new-hold states', Q[nh].mean(0) if nh.sum() else None)
    lh = m & (gs != g) & (g == ph)
    print(f'   lost holds {lh.sum()} by current phase', np.bincount(ph[lh], minlength=4), ' now picks', np.bincount(gs[lh], minlength=4))
    print('   mean halting per phase in lost-hold states', Q[lh].mean(0) if lh.sum() else None)
    sw = m & (gs != g) & (gs != ph) & (g != ph)
    T = np.zeros((4,4), int)
    for a, b in zip(g[sw], gs[sw]): T[a, b] += 1
    print('   switch-target changes (row old pick, col new pick)\n', T)
