"""check.py: offline checks of a patched network (forward passes only).
    python work/check.py work/<file>.npz [reference.npz]"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
names = tools.names(); I = {n: i for i, n in enumerate(names)}
serve = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
lanes = ['N0', 'N1', 'N2', 'E0', 'E1', 'E2', 'S0', 'S1', 'S2', 'W0', 'W1']
w = tools.load(sys.argv[1])
ref = tools.load(sys.argv[2] if len(sys.argv) > 2 else 'work/wA.npz')
print('edit size', tools.edit_size(w, tools.load('start_weights.npz')))
S = tools.states(); X = S['obs'].astype(float); c = S['controller']; ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)
short = X[:, 45] < 0.45
g = tools.greedy(w, X); gr = tools.greedy(ref, X)
for ci, cn in enumerate(['start', 'fixed', 'act']):
    m = (c == ci) & ok & short
    print(f'{cn}: ok & green<27s states {m.sum()}, agree with ref {(g[m] == gr[m]).mean():.4f}; acts', np.bincount(g[m], minlength=4))
# deadlock states from the recorded start-network episodes
m = (c == 0) & (S['episode'] == 3) & (S['time_s'] > 900) & (S['time_s'] < 3600)
print('deadlock states -> acts', np.bincount(g[m], minlength=4))
# idle sweeps
for p in range(4):
    row = []
    for s in [2, 12, 22, 27, 32, 37, 47, 60]:
        z = np.zeros(46); z[40 + p] = 1; z[44] = 1 if s >= 10 else 0; z[45] = min(s / 60, 1)
        row.append(f'{s}s:{tools.greedy(w, z)[0]}')
    print('idle phase', p, ' '.join(row))
# single-lane stuck search: lane not served by p, network keeps p
stuck = []
for p in range(4):
    for ln in lanes:
        if ln in serve[p]: continue
        for k in [1, 2, 3]:
            for tg in [0.2, 0.5, 1.0]:
                z = np.zeros(46); z[40 + p] = 1; z[44] = 1; z[45] = tg
                z[I['q_' + ln]] = min(k / 2.99, 1); z[I['d_' + ln]] = z[I['q_' + ln]]
                if tools.greedy(w, z)[0] == p: stuck.append(f'p{p}:{ln}x{k}@{int(tg*60)}s')
print('single-lane holds', len(stuck), stuck)
print(tools.trainability(w))
