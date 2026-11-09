import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
SL = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
UPx = {0: [0,4], 1: [], 2: [2,6], 3: []}
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
Ds = np.stack([d[:, SL[p]].sum(1) for p in range(4)], 1)
g = tools.greedy(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
age = np.round(obs[:, 45]*60)
I = np.arange(len(t))
for c in range(3):
    m0 = (ctl == c)
    print("ctl", c)
    for p in range(4):
        for a in [2, 7, 12, 17]:
            mm = m0 & (ph == p) & (age == a)
            if mm.sum() < 10: continue
            print(f"  ph{p} age{a:2.0f} n={mm.sum():4d} stopline veh on phase lanes={Ds[mm, p].mean():.2f} halted={np.stack([q[:, SL[k]].sum(1) for k in range(4)],1)[mm, p].mean():.2f}  moving-on-stopline={(Ds[mm,p]-np.stack([q[:, SL[k]].sum(1) for k in range(4)],1)[mm,p]).mean():.2f}")
