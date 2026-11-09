import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
for c in range(3):
    m = ctl == c
    print(f"ctl {c}: halted stopline {q[m].sum(1).mean():.2f}  upstream {qu[m].sum(1).mean():.2f}  per-up-lane {qu[m].mean(0)}")
    print(f"        vehicles stopline {d[m].sum(1).mean():.2f}  upstream {du[m].sum(1).mean():.2f}")
    # per episode-level: mean total vehicles (proxy TTS) and duration
    for e in range(6):
        me = m & (ep == e)
        print(f"   ep{e}: steps {me.sum()}  mean veh {(d[me].sum(1)+du[me].sum(1)).mean():.1f}  veh-h proxy {(d[me].sum(1)+du[me].sum(1)).sum()*5/3600:.1f} halted-h {(q[me].sum(1)+qu[me].sum(1)).sum()*5/3600:.1f}")
