import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); A = S['act']; E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
# red time per phase between services for start network
for c in range(3):
    m = C == c
    reds = {p: [] for p in range(4)}
    for e in np.unique(E[m]):
        idx = np.where(m & (E == e))[0]
        idx = idx[T[idx] < 3600]
        last_end = {p: None for p in range(4)}
        prev = None
        for i in idx:
            p = ph[i]
            if prev is not None and p != prev:
                last_end[prev] = T[i]
                if last_end[p] is not None:
                    reds[p].append(T[i] - last_end[p])
            prev = p
    print(S['controller_names'][c], {p: (len(r), np.mean(r).round(1), np.quantile(r, .9), max(r)) for p, r in reds.items()})
# cycle composition: how many phases served per ~ "cycle"
m = (C == 0)
g = tools.greedy(w, X)
# disagreement samples with max-queue
q = X[:, :12] * 3
Q = np.stack([q[:, SERVE[p]].sum(1) for p in range(4)], 1)
mm = np.where(m & ok)[0]
rng = np.random.default_rng(0)
for i in rng.choice(mm, 25, replace=False):
    print(T[i], 'cur', ph[i], 'pick', g[i], 'Q', Q[i], 'q', X[i, :12].round(1)*3, 'uq', (X[i, 24:32]*34).round(0), 'ud', (X[i,32:40]*34).round(0))
