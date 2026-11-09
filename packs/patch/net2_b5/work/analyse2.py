import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); A = S['act']; E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5; tg = np.round(X[:, 45] * 60)
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g = tools.greedy(w, X)
lg = tools.forward(w, X)
# vehicle counts: stop-line cap = 22.4/7.5 = 2.987 -> 3 ; upstream 256/7.5=34.1
q_veh = X[:, :12] * 3; d_veh = X[:, 12:24] * 3
uq = X[:, 24:32] * 34.13; ud = X[:, 32:40] * 34.13
def pq(arr):
    return np.stack([arr[:, SERVE[p]].sum(1) for p in range(4)], 1)
Q = pq(q_veh); D = pq(d_veh)
for c in range(3):
    m = (C == c) & ok
    gg = g[m]; Qm = Q[m].copy(); pm = ph[m]
    Qm[np.arange(len(pm)), pm] = -1
    mq = Qm.argmax(1)
    print(S['controller_names'][c], 'ok states', m.sum(), 'net picks current', (gg == pm).mean().round(3),
          'agree w maxQ(other)', (gg == mq).mean().round(3))
    # when disagree, queue picked vs max
    dis = (gg != mq) & (gg != pm)
    print('   disagreements', dis.sum(), 'mean Q picked', Qm[dis, gg[dis]].mean().round(2), 'mean Q max', Qm[dis, mq[dis]].mean().round(2))
# the ok-state logits margin for start network
m = (C == 0) & ok
srt = np.sort(lg[m], 1)
print('ok state logit gap quantiles', np.quantile(srt[:, -1] - srt[:, -2], [.05, .25, .5, .75]).round(2))
# upstream queue presence
for c in range(3):
    m = C == c
    print(S['controller_names'][c], 'frac states with any upstream queue', (X[m, 24:32].max(1) > 0).mean().round(3),
          'mean total halting veh', (q_veh[m].sum(1) + uq[m].sum(1)).mean().round(2))
# episode lengths
for c in range(3):
    m = C == c
    print(S['controller_names'][c], [int(T[m & (E == e)].max()) for e in np.unique(E[m])])
