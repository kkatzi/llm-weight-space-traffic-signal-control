import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
names = tools.names()
ph = X[:, 40:44].argmax(1)
# per controller-episode: green durations per phase, number of switches, length
for ci in range(3):
    print('controller', S['controller_names'][ci])
    m = c == ci
    for e in np.unique(ep[m]):
        mm = m & (ep == e)
        idx = np.where(mm)[0]
        p = ph[idx]
        sw = (np.diff(p) != 0).sum()
        # green durations: runs of p
        runs = []
        cur = p[0]; L = 1
        for v in p[1:]:
            if v == cur: L += 1
            else: runs.append((cur, L)); cur = v; L = 1
        runs.append((cur, L))
        dur = {k: np.mean([l*5 for (q,l) in runs if q == k]) if any(q==k for q,_ in runs) else 0 for k in range(4)}
        print(f'  ep {e} len {len(idx)} ({t[idx].max()} s) switches {sw}  mean run(s) per phase', {k: round(v,1) for k,v in dur.items()})
