import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; ep = S["episode"]; t = S["time_s"]
nm = tools.names()
for ci, cn in enumerate(S["controller_names"]):
    for e in np.unique(ep[c == ci]):
        m = (c == ci) & (ep == e)
        tt = t[m]; Xe = X[m]
        last = Xe[-1]
        nz = {nm[i]: round(float(v), 3) for i, v in enumerate(last[:40]) if v > 0}
        # find the last time any vehicle count changed
        tot = Xe[:, 12:24].sum(1) + Xe[:, 32:40].sum(1)
        print(f"{cn:13s} ep {e}: n={m.sum()} t_end={tt[-1]} last phase {last[40:44].argmax()} tg={last[45]:.2f} nonzero={nz}")
