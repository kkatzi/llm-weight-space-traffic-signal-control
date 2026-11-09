import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]; ep = S["episode"]; t = S["time_s"]; A = S["act"]
nm = tools.names()
def show(ci, e, t0, t1, step=1):
    m = np.where((c == ci) & (ep == e) & (t >= t0) & (t <= t1))[0]
    for i in m[::step]:
        x = X[i]
        nz = " ".join(f"{nm[j]}={x[j]:.2f}" for j in range(40) if x[j] > 0)
        print(f"  t={t[i]} ph={x[40:44].argmax()} ok={int(x[44])} tg={x[45]:.2f} act={A[i]} | {nz}")
for ci, e in [(0, 1), (0, 5), (2, 5)]:
    m = (c == ci) & (ep == e)
    tot = X[m][:, 12:24].sum(1) + X[m][:, 32:40].sum(1)
    tt = t[m]
    # first time after 3600 where state becomes constant-ish
    print(S["controller_names"][ci], e)
    show(ci, e, 3400, 3900, 3)
    print("  ...")
    show(ci, e, 5300, 5400, 4)
