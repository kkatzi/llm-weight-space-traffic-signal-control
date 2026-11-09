import sys; sys.path.insert(0, '.')
import numpy as np, tools
from collections import Counter
np.set_printoptions(precision=2, suppress=True, linewidth=220)
exec(open('work/explore24.py').read().split("g = tools.greedy")[0])
exec(open('work/explore21.py').read().split("X = obs")[0].split("S = tools.states()")[1].replace("obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']", "") if False else "")
N_ = tools.names(); ix = {n: i for i, n in enumerate(N_)}
def perm_for(mapping):
    out = []
    for n in N_:
        if n.startswith(("q_", "d_")):
            pre, lane = n.split("_"); out.append(ix[f"{pre}_{mapping[lane[0]]}{lane[1:]}"])
        else: out.append(ix[n])
    return np.array(out)
R = perm_for({"N": "S", "S": "N", "E": "W", "W": "E"})
lg = tools.forward(w, obs); lgR = tools.forward(w, obs[:, R]); g = lg.argmax(1); gR = lgR.argmax(1)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
H = W(obs, near=0.0); HR = W(obs[:, R], near=0.0)
G = 10
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
    l2 = lg + alpha*(H - G*lam); l2R = lgR + alpha*(HR - G*lam)
    g2 = l2.argmax(1); g2R = l2R.argmax(1)
    s = []
    for c in range(3):
        m = ok & (ctl == c)
        s.append(f"ctl{c} changed={(g2[m]!=g[m]).mean():.3f} rotdis={(g2[m]!=g2R[m]).mean():.3f}")
    print(f"alpha={alpha}: ", "  ".join(s))
