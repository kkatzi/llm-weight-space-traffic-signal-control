import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; act = S['act']; t = S['time_s']
names = tools.names()
a = tools.greedy(w, X)
cap_sl = 22.4/7.5; cap_up = 256/7.5
for c in range(3):
  for e in np.unique(ep[ctrl==c]):
    mm = (ctrl==c)&(ep==e)
    Xe = X[mm]; te=t[mm]
    ph = Xe[:,40:44].argmax(1)
    # vehicles approx in system: density*cap
    nveh = Xe[:,12:24].sum(1)*cap_sl + Xe[:,32:40].sum(1)*cap_up
    # green durations
    durs = {p:[] for p in range(4)}
    cur=ph[0]; start=te[0]
    for i in range(1,len(ph)):
        if ph[i]!=cur:
            durs[cur].append(te[i]-start); cur=ph[i]; start=te[i]
    print(c, e, len(te), te.max(), 'mean veh %.1f max %.1f'%(nveh.mean(), nveh.max()),
          'nswitch', sum(len(v) for v in durs.values()),
          ' '.join('p%d:n%d,m%.0f'%(p,len(d),np.mean(d) if d else 0) for p,d in durs.items()),
          'frac up sat', (Xe[:,32:40]>=0.99).mean().round(3))
