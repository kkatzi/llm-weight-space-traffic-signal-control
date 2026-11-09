import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; ep = S["episode"]; t = S["time_s"]
ph = X[:, 40:44].argmax(1)
sn = c == 0
tot = X[:, 12:24].sum(1) * 3 + X[:, 32:40].sum(1) * 34  # approx vehicle count
for k in range(4):
    m = sn & (ph == k)
    tg = X[m, 45]
    print(f"phase {k}: n={m.sum()} t_green quantiles", np.percentile(tg * 60, [50, 75, 90, 95, 99, 100]).round(0),
          " n(t_green=1)", int((tg >= 0.999).sum()), " of which t<3600:", int(((tg >= 0.999) & (t[m] < 3600)).sum()))
m = sn & (X[:, 45] >= 0.999) & (t < 3600)
print("t_green=1 during demand: phase", np.bincount(ph[m], minlength=4), " approx veh count quantiles", np.percentile(tot[m], [0, 25, 50, 75, 100]).round(1))
for i in np.where(m)[0][:40]:
    x = X[i]; nm = tools.names()
    print(f"  ep{ep[i]} t={t[i]} ph={ph[i]} act={S['act'][i]} | " + " ".join(f"{nm[j]}={x[j]:.2f}" for j in range(40) if x[j] > 0))
