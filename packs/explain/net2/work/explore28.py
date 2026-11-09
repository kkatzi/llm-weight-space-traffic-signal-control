from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
Gs = np.mean([GP[p] for p in range(4)], 0)
c = (Gs[:, 0:12] + Gs[:, 12:24]) / 2 / 3     # per unit of v = 3(q+d)
c = c - c.mean(0, keepdims=True)
print("coefficient per unit of lane load v=3(q+d) (halting car = 2 units, moving = 1)")
print("      " + " ".join(f"{l:>5s}" for l in LANES))
for o in range(4): print(f"L{o}    " + " ".join(f"{c[o, i]:5.2f}" for i in range(12)))
# intercepts per current phase: value of linear model at x with zero stop-line load, other features at XB[p]
for p in range(4):
    G = Gs.copy(); cc = (G[:, 0:12] + G[:, 12:24]) / 2; G[:, 0:12] = cc; G[:, 12:24] = cc
    G[:, 24:] = 0
    Xz = XB[p].copy()
    icpt = LB[p] - (XB[p] @ G.T)  # model value at zero stop-line load
    print("phase", p, "intercepts", icpt.round(2), "rel. to mean of candidates:", (icpt - np.delete(icpt, p).mean()).round(2))
