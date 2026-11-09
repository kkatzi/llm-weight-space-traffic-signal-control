import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
thr_ph = {0: 0, 1: 2, 2: 0, 3: 2}; left_ph = {0: 1, 1: 3, 2: 1, 3: 3}
for c in range(3):
    print("=== ctl", c)
    for a, nm in enumerate("NESW"):
        pocket_full = obs[:, 3*a+2] > 0.99
        up1q = qu[:, 2*a+1]
        lane1 = q[:, 3*a+1]
        # blocked after own through green: current phase is through for this axis, lane1 stop-line almost empty, up1 queue>=2
        blk = ok & (ctl == c) & pocket_full & (up1q >= 2) & (ph == thr_ph[a]) & (lane1 < 1.5)
        if blk.sum():
            print(f"  {nm}: blocked-after-through states {blk.sum()}, net picks {np.bincount(g[blk], minlength=4)}, mean P(left)={P[blk, left_ph[a]].mean():.2f}")
        blk2 = ok & (ctl == c) & pocket_full & (up1q >= 2)
        print(f"  {nm}: pocket full & up1 queue>=2 states {blk2.sum()}, net picks {np.bincount(g[blk2], minlength=4)}; cur ph {np.bincount(ph[blk2], minlength=4)}")
