import sys; sys.path.insert(0, '.')
import numpy as np, tools
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
d = obs[:, 12:24] * CS; du = obs[:, 32:40] * CU
q = obs[:, 0:12] * CS; qu = obs[:, 24:32] * CU
A = "NESW"
Napp = np.stack([d[:, 3*a:3*a+3].sum(1) + du[:, 2*a:2*a+2].sum(1) for a in range(4)], 1)
Qapp = np.stack([q[:, 3*a:3*a+3].sum(1) + qu[:, 2*a:2*a+2].sum(1) for a in range(4)], 1)
nxt = np.r_[np.arange(1, len(t)), 0]
valid = (ep[nxt] == ep) & (ctl[nxt] == ctl) & (t[nxt] - t == 5)
ph = obs[:, 40:44].argmax(1); age = np.round(obs[:, 45]*60).astype(int); ok = obs[:, 44] > .5
sw = (act != ph) & ok
served = {0: [0, 2], 1: [0, 2], 2: [1, 3], 3: [1, 3]}
red = np.ones((len(t), 4), bool)
for p in range(4):
    for a in served[p]: red[ph == p, a] = False
dN = Napp[nxt] - Napp
# arrival rate per (ctl, ep, approach, 15min bin)
bins = np.minimum(t // 900, 4)
lam = np.zeros((len(t), 4))
for c in range(3):
    for e in range(6):
        for b in range(5):
            m = (ctl == c) & (ep == e) & (bins == b) & valid
            for a in range(4):
                mm = m & red[:, a] & ~sw
                lam[m, a] = dN[mm, a].mean() if mm.sum() > 5 else np.nan
dep = lam - dN   # departures in the next 5 s per approach
for c in range(3):
    print("=== controller", c)
    for p in range(4):
        for ag in [2, 7, 12, 17, 22, 27]:
            m = valid & (ctl == c) & (ph == p) & (age == ag) & ~sw
            if m.sum() < 15: continue
            a0, a1 = served[p]
            dd = dep[m][:, [a0, a1]].sum(1)
            Q = Qapp[m][:, [a0, a1]].sum(1)
            hi = Q > 6
            print(f"  ph{p} age{ag:2d} n={m.sum():4d} dep/5s={np.nanmean(dd):5.2f}  (queue>6: n={hi.sum():4d} dep={np.nanmean(dd[hi]) if hi.sum() else np.nan:5.2f})  meanQ={Q.mean():5.1f}")
        # switching interval: yellow
        m = valid & (ctl == c) & (ph == p) & sw
        a0, a1 = served[p]
        print(f"  ph{p} SWITCH n={m.sum()} dep of old phase approaches in yellow interval={np.nanmean(dep[m][:, [a0,a1]].sum(1)):5.2f}")
