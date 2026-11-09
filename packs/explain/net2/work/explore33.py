from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
GR = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
OWN = {l: g for g, ls in GR.items() for l in ls}
Gs = np.mean([GP[p] for p in range(4)], 0)
c = (Gs[:, 0:12] + Gs[:, 12:24]) / 2 / 3; c = c - c.mean(0, keepdims=True)
rw = [0.25, 1, 1, 0.75, 1, 1, 0.75, 0.75, 1, 0.75, 1.5, 1]
M = np.array([[1.0, 0, 0, 0.2],[0, 1.35, 0, 0],[0.5, -0.55, 1.25, 0],[-0.25, -0.15, 0.45, 1.55]]) * 1.13
cm = np.array([[M[o, OWN[l]] * rw[l] for l in range(12)] for o in range(4)]); cm = cm - cm.mean(0, keepdims=True)
print("lane-level (network Taylor, centered)"); print("      " + " ".join(f"{l:>5s}" for l in LANES))
for o in range(4): print(f"L{o}    " + " ".join(f"{c[o, i]:5.2f}" for i in range(12)))
print("group model implied (centered)")
for o in range(4): print(f"L{o}    " + " ".join(f"{cm[o, i]:5.2f}" for i in range(12)))
print("residual network - group model")
for o in range(4): print(f"L{o}    " + " ".join(f"{c[o, i]-cm[o, i]:5.2f}" for i in range(12)))
# typical spread of each lane load
Vv = np.stack([v(X[OK], n) for n in LANES], 1)
print("std of lane loads", Vv.std(0).round(2))
