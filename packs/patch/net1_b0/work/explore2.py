import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'], S['episode'], S['time_s'], S['controller']
names = tools.names()
m = (ctl == 0)
for e in range(6):
    me = m & (ep == e)
    o = obs[me]; a = act[me]; tt = t[me]
    ph = o[:, 40:44].argmax(1)
    print("ep", e, "n", me.sum(), "t range", tt.min(), tt.max())
    # green durations: track phase runs
    runs = []
    cur = ph[0]; start = 0
    for i in range(1, len(ph)):
        if ph[i] != cur:
            runs.append((cur, i - start)); cur = ph[i]; start = i
    runs.append((cur, len(ph) - start))
    # sequence transitions
    from collections import Counter
    trans = Counter((runs[i][0], runs[i+1][0]) for i in range(len(runs)-1))
    dur = {p: np.array([d for (q, d) in runs if q == p]) for p in range(4)}
    print("  runs", len(runs), {p: (len(dur[p]), round(dur[p].mean()*5,1) if len(dur[p]) else 0, dur[p].max()*5 if len(dur[p]) else 0) for p in range(4)})
    print("  trans", sorted(trans.items()))
