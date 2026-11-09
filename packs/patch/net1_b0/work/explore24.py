import sys; sys.path.insert(0, '.')
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=1, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'].astype(np.float64), S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UPx = {0: [0,4], 1: [], 2: [2,6], 3: []}; UPs = {0: [1,5], 1: [1,5], 2: [3,7], 3: [3,7]}
lam = np.array([0.175, 0.075, 0.29, 0.125])
def W(o, near=0.25):
    q = o[:, 0:12]*CS; d = o[:, 12:24]*CS; qu = o[:, 24:32]*CU; du = o[:, 32:40]*CU
    out = []
    for p in range(4):
        v = d[:, SL[p]].sum(1) + qu[:, UPx[p]].sum(1) + near*(du-qu)[:, UPx[p]].sum(1)
        share = 0.5
        v = v + share*(qu[:, UPs[p]].sum(1) + near*(du-qu)[:, UPs[p]].sum(1))
        out.append(v)
    return np.stack(out, 1)
Wp = W(obs) + 7.5*lam
g = tools.greedy(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
for c in range(3):
    m = ok & (ctl == c) & (g != ph)
    sm = Wp.copy(); sm[I, ph] = -1e9  # switching only (exclude current)
    s = sm.argmax(1)
    print(f"ctl{c}: switch decisions {m.sum()}, agree with Smith index {(s[m]==g[m]).mean():.2f}")
    cnt = Counter((g[i], s[i]) for i in np.where(m & (s != g))[0])
    print("   (net, smith) disagreements:", cnt.most_common(12))
    ii = np.where(m & (s != g))[0]
    print("   mean W(net choice)", Wp[ii, g[ii]].mean().round(2), " mean W(smith choice)", Wp[ii, s[ii]].mean().round(2))
