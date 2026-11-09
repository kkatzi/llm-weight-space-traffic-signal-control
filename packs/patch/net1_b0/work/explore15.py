import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=1, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
obs, act, ep, t, ctl = S['obs'], S['act'].astype(int), S['episode'], S['time_s'], S['controller']
CS, CU = 22.4/7.5, 256/7.5
q = obs[:, 0:12]*CS; d = obs[:, 12:24]*CS; qu = obs[:, 24:32]*CU; du = obs[:, 32:40]*CU
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
def show(e, t0, t1, c=0):
    m = (ctl == c) & (ep == e) & (t >= t0) & (t <= t1)
    print(f"--- ctl{c} ep{e} {t0}-{t1}   cols: ph age | act | qE(0,1,2) dE(0,1,2) | quE0u quE1u duE0u duE1u | qW.. ")
    for i in np.where(m)[0]:
        print(f"t={t[i]:4d} ph={ph[i]} age={obs[i,45]*60:4.0f} act={act[i]} P={P[i].round(2)} | E q{q[i,3:6]} d{d[i,3:6]} up q{qu[i,2:4]} d{du[i,2:4]} | W q{q[i,9:12]} d{d[i,9:12]} up q{qu[i,6:8]} d{du[i,6:8]} | N q{q[i,0:3]} S q{q[i,6:9]}")
show(5, 1440, 1530)
