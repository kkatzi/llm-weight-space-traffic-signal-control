import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5
tg = X[:, 45]*60
for ci in range(3):
    m = c == ci
    T = np.zeros((4,4), int)
    for e in np.unique(ep[m]):
        idx = np.where(m & (ep == e))[0]
        p = ph[idx]
        for a_, b_ in zip(p[:-1], p[1:]):
            if a_ != b_: T[a_, b_] += 1
    print(S['controller_names'][ci], 'transition counts (from row to col)\n', T)
    # green time at switch distribution
    sw = m & ok & (A != ph)
    print(' t_green at honoured switch:', np.unique(np.round(tg[sw]), return_counts=True))
    hold = m & ok & (A == ph)
    print(' holds when ok:', hold.sum(), ' of ok', (m&ok).sum())
    # network greedy on these states
    g = tools.greedy(w, X[m])
    okm = ok[m]
    print(' network would hold when ok:', (g[okm]==ph[m][okm]).mean())
