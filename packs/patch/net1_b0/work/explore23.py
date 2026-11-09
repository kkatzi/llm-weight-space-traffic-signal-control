import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=1, suppress=True, linewidth=220)
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
ph = obs[:, 40:44].argmax(1)
for c in range(3):
    gaps = {p: [] for p in range(4)}
    for e in range(6):
        m = (ctl == c) & (ep == e) & (t < 3600)
        pp = ph[m]; tt = t[m]
        last_end = {p: None for p in range(4)}
        for i in range(1, len(pp)):
            if pp[i] != pp[i-1]:
                last_end[pp[i-1]] = tt[i]
                if last_end[pp[i]] is not None:
                    gaps[pp[i]].append(tt[i] - last_end[pp[i]])
    print("ctl", c, {p: (len(g), np.mean(g).round(1), np.percentile(g, 90), np.max(g)) for p, g in gaps.items()})
