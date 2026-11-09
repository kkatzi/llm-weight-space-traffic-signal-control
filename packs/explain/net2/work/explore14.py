from common import *
np.set_printoptions(linewidth=220, precision=2, suppress=True)
m = OK & (CTRL == 0)
Xb = X[m]; Lb = LG[m]; pb = PH[m]
print("finite difference: add one halting vehicle (q,d += 1/3) on lane; mean change in logit_j minus mean over 4 logits")
print("lane  n   dL0   dL1   dL2   dL3   | moving vehicle (d only) dL0..3")
for i, n in enumerate(LANES):
    sel = Xb[:, 12+i] < 0.9
    X1 = Xb[sel].copy(); X1[:, i] = np.minimum(X1[:, i] + 1/3, 1); X1[:, 12+i] = np.minimum(X1[:, 12+i] + 1/3, 1)
    d = tools.forward(w, X1) - Lb[sel]; d = d - d.mean(1, keepdims=True)
    X2 = Xb[sel].copy(); X2[:, 12+i] = np.minimum(X2[:, 12+i] + 1/3, 1)
    d2 = tools.forward(w, X2) - Lb[sel]; d2 = d2 - d2.mean(1, keepdims=True)
    print(f"{n:4s} {sel.sum():4d} " + " ".join(f"{x:5.2f}" for x in d.mean(0)) + "   | " + " ".join(f"{x:5.2f}" for x in d2.mean(0)))
up = ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u']
print("upstream: add 3 moving vehicles (d += 3/34)")
for k, n in enumerate(up):
    X1 = Xb.copy(); X1[:, 32+k] = np.minimum(X1[:, 32+k] + 3/34.13, 1)
    d = tools.forward(w, X1) - Lb; d = d - d.mean(1, keepdims=True)
    X2 = Xb.copy(); X2[:, 24+k] = np.minimum(X2[:, 24+k] + 3/34.13, 1); X2[:, 32+k] = np.minimum(X2[:, 32+k] + 3/34.13, 1)
    d2 = tools.forward(w, X2) - Lb; d2 = d2 - d2.mean(1, keepdims=True)
    print(f"{n:4s} " + " ".join(f"{x:5.2f}" for x in d.mean(0)) + "   | 3 halting: " + " ".join(f"{x:5.2f}" for x in d2.mean(0)))
print("t_green +5 s")
X1 = Xb.copy(); X1[:, 45] += 5/60
d = tools.forward(w, X1) - Lb; d = d - d.mean(1, keepdims=True); print(d.mean(0))
