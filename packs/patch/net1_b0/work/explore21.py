import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
N = tools.names(); ix = {n: i for i, n in enumerate(N)}
def perm_for(mapping):
    p = np.arange(46)
    for i, n in enumerate(N):
        for a, b in mapping.items():
            pass
    out = []
    for n in N:
        if n.startswith(("q_", "d_")):
            pre, lane = n.split("_")
            ap = lane[0]; rest = lane[1:]
            out.append(ix[f"{pre}_{mapping[ap]}{rest}"])
        else:
            out.append(ix[n])
    return np.array(out)
rot180 = perm_for({"N": "S", "S": "N", "E": "W", "W": "E"})
X = obs.astype(np.float64)
Xr = X[:, rot180]
g = tools.greedy(w, X); gr = tools.greedy(w, Xr)
ok = X[:, 44] > .5
lg = tools.forward(w, X); srt = np.sort(lg, 1); gap = srt[:, -1] - srt[:, -2]
for c in range(3):
    m = ok & (ctl == c)
    print(f"ctl{c}: honoured {m.sum()}, 180-rotation disagreement {(g[m] != gr[m]).mean():.3f}; among gap>1: {(g[m & (gap>1)] != gr[m & (gap>1)]).mean():.3f}")
# Also N-S swap only (mirror) and E-W swap only
for nm, mp in [("NS swap", {"N": "S", "S": "N", "E": "E", "W": "W"}), ("EW swap", {"N": "N", "S": "S", "E": "W", "W": "E"})]:
    pp = perm_for(mp); g2 = tools.greedy(w, X[:, pp])
    m = ok & (ctl == 0)
    print(nm, "disagreement", (g[m] != g2[m]).mean())
