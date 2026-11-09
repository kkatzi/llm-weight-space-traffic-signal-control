import sys; sys.path.insert(0, '.')
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'].astype(np.float64), S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UPx = {0: [0,4], 1: [], 2: [2,6], 3: []}; UPs = {0: [1,5], 1: [1,5], 2: [3,7], 3: [3,7]}
lam = np.array([0.175, 0.075, 0.2917, 0.125])
def Hmat(o):
    d = o[:, 12:24]*CS; qu = o[:, 24:32]*CU
    return np.stack([d[:, SL[p]].sum(1) + qu[:, UPx[p]].sum(1) + 0.5*qu[:, UPs[p]].sum(1) for p in range(4)], 1)
H = Hmat(obs)
lg = tools.forward(w, obs); g = lg.argmax(1)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
print("H quantiles per phase (ok states, ctl0):", np.quantile(H[ok & (ctl==0)], [.5, .9, .99, 1], axis=0))
sA = (H[:, 0] - H[:, 2])/2; sB = (H[:, 1] - H[:, 3])/2; sC = (H[:, 0] + H[:, 2] - H[:, 1] - H[:, 3])/4
for nm, s in [("sA", sA), ("sB", sB), ("sC", sC)]:
    print(nm, "quantiles abs", np.quantile(np.abs(s[ok]), [.5, .9, .99, 1]))
eA = np.array([1, 0, -1, 0]); eB = np.array([0, 1, 0, -1]); eC = np.array([1, -1, 1, -1])
G = 10
for alpha in [0.3, 0.5, 0.75, 1.0]:
    for chans in ["ABC", "AB", "AC", "BC"]:
        off = -alpha*G*lam[None, :] + 0
        for ch in chans:
            s, e = {"A": (sA, eA), "B": (sB, eB), "C": (sC, eC)}[ch]
            off = off + alpha*s[:, None]*e[None, :]
        g2 = (lg + off).argmax(1)
        m = ok & (ctl == 0)
        ch_ = m & (g2 != g)
        gapn = lg[I, g] - lg[I, g2]
        print(f"alpha={alpha} chans={chans}: changed {ch_.sum()} ({ch_.sum()/m.sum():.3f}); median net gap of changed {np.median(gapn[ch_]) if ch_.sum() else 0:.2f}; pairs {Counter(zip(g[ch_], g2[ch_])).most_common(5)}")
