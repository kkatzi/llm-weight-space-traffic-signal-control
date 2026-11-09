import sys; sys.path.insert(0, '.')
import numpy as np, tools
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
# approach vehicle counts
def counts(o):
    d = o[:, 12:24] * CS; du = o[:, 32:40] * CU
    out = {}
    for a, nm in enumerate("NESW"):
        out[nm+"_thr"] = d[:, 3*a] + d[:, 3*a+1]
        out[nm+"_left"] = d[:, 3*a+2]
        out[nm+"_up"] = du[:, 2*a] + du[:, 2*a+1]
        out[nm] = d[:, 3*a:3*a+3].sum(1) + du[:, 2*a:2*a+2].sum(1)
    return out
C = counts(obs)
print("upstream density==1 share:", (obs[:, 32:40] >= 0.999).mean(0))
print("stopline density==1 share:", (obs[:, 12:24] >= 0.999).mean(0).round(2))
# consecutive pairs
nxt = np.r_[np.arange(1, len(t)), 0]
valid = (ep[nxt] == ep) & (ctl[nxt] == ctl) & (t[nxt] - t == 5)
print("valid pairs", valid.sum(), "of", len(t))
ph = obs[:, 40:44].argmax(1); age = np.round(obs[:, 45]*60).astype(int); ok = obs[:, 44] > .5
# effective action: switch happens if act != ph and ok
sw = (act != ph) & ok
# arrivals estimate per approach: intervals where approach is not served and no switch involving it
serve = {0: "NS", 1: "NS", 2: "EW", 3: "EW"}
for c in range(3):
    print("=== controller", c)
    for a in "NESW":
        m = valid & (ctl == c) & ~sw & np.array([a not in serve[p] for p in ph])
        dN = C[a][nxt] - C[a]
        print(f"  arrivals {a}: {dN[m].mean()*720:.0f} veh/h (n={m.sum()})")
