import numpy as np, sys
sys.path.insert(0, '.')
import tools
w = tools.load('start_weights.npz'); S = tools.states()
X = S['obs'].astype(np.float64); A = S['act']; ep = S['episode']; t = S['time_s']; c = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
capS = 22.4/7.5; capU = 256/7.5
PL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
np.set_printoptions(precision=2, suppress=True, linewidth=150)
for ci in range(3):
    m = c == ci
    reds = {p: [] for p in range(4)}
    qwait = {p: [] for p in range(4)}  # queue-weighted
    for e in np.unique(ep[m]):
        idx = np.where(m & (ep == e))[0]
        p_ = ph[idx]; tt = t[idx]
        last = {p: None for p in range(4)}
        # red interval of phase p: from end of its green to start of next green
        runs = []
        s = 0
        for i in range(1, len(p_)+1):
            if i == len(p_) or p_[i] != p_[s]:
                runs.append((p_[s], tt[s], tt[i-1]))
                s = i
        lastend = {}
        for (p, st, en) in runs:
            if p in lastend:
                reds[p].append(st - lastend[p])
            lastend[p] = en
    print(S['controller_names'][ci])
    for p in range(4):
        r = np.array(reds[p])
        print(f'  phase {p}: n={len(r)} red-gap mean {r.mean():.1f} pct50/90/99/max', np.percentile(r, [50,90,99,100]))
