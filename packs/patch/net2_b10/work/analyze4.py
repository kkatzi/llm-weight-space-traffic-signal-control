import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=180)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); act = S["act"]; ctrl = S["controller"]; ep = S["episode"]; t = S["time_s"]
names = tools.names()
cap = 22.4/7.5; capu = 256/7.5
print("unique q values lane N0:", np.unique(np.round(X[:,0]*cap,2))[:10])
print("unique qu values:", np.unique(np.round(X[:,24]*capu,2))[:10])
for c in range(3):
    m = ctrl == c
    print("ctrl", S["controller_names"][c])
    print("  frac q_up>0:", (X[m, 24:32] > 0).mean(0))
    print("  mean q_up veh:", (X[m, 24:32]*capu).mean(0))
    print("  max q_up veh:", (X[m, 24:32]*capu).max(0))
    print("  frac stopline lane full queue(q>=0.99):", (X[m, 0:12] > 0.99).mean(0))
# per-episode summary of network episodes: time, total vehicles queued
for c in range(3):
    for e in range(6):
        m = (ctrl == c) & (ep == e)
        tot = (X[m, 12:24].sum(1)*cap + X[m, 32:40].sum(1)*capu)
        totq = (X[m, 0:12].sum(1)*cap + X[m, 24:32].sum(1)*capu)
        print(c, e, "n", m.sum(), "tmax", t[m].max(), "mean veh on approaches", tot.mean().round(1), "mean halted", totq.mean().round(1), "peak veh", tot.max().round(1))
