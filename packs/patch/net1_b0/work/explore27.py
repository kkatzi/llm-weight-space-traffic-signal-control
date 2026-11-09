import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=1, suppress=True, linewidth=220)
exec(open('work/explore24.py').read().split("g = tools.greedy")[0])
g = tools.greedy(w, obs); P = tools.probs(w, obs)
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
W0 = W(obs, near=0.25)   # without lambda term
for c in range(3):
    m = ok & (ctl == c) & (t < 3550) & (t > 30)
    hold = m & (g == ph)
    bad_hold = hold & (W0[I, ph] < 0.6) & (np.where(np.eye(4)[ph] > 0, -1, W0).max(1) >= 2)
    bad_sw = m & (g != ph) & (W0[I, g] < 0.6) & (np.where(np.eye(4)[g] > 0, -1, W0).max(1) >= 2)
    print(f"ctl{c}: honoured {m.sum()} holds {hold.sum()} bad holds {bad_hold.sum()} bad switches {bad_sw.sum()}")
    for i in np.where(bad_hold | bad_sw)[0][:20]:
        print(f"   ep{ep[i]} t={t[i]} ph={ph[i]} net={g[i]} P={P[i].round(2)} W={W0[i]} q={q[i]} d={d[i]} qu={qu[i]} du={du[i]}")
