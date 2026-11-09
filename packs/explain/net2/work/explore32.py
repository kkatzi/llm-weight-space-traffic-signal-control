from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
V = {n: v(X, n) for n in LANES}
A = 0.25*V['N0'] + V['N1'] + 0.75*V['S0'] + 0.75*V['S1']; B = V['N2'] + V['S2']
C = 0.75*V['E0'] + V['E1'] + 0.75*V['W0'] + 1.5*V['W1']; D = V['E2'] + V['W2']
Ttot = sum(V[n] for n in LANES)
up = X[:, 32:40].sum(1)*34.13; upq = X[:, 24:32].sum(1)*34.13
tg = X[:, 45]*60
m3 = OK & (PH == 3)
for lab, m in [("stay3", m3 & (A == 3)), ("leave3", m3 & (A != 3))]:
    pass
stay = m3 & (np.argmax(LG, 1) == 3); leave = m3 & ~stay
for lab, m in [("stay3", stay), ("leave3", leave)]:
    print(lab, m.sum(), "A", A[m].mean().round(2), "B", B[m].mean().round(2), "C", C[m].mean().round(2), "D", D[m].mean().round(2),
          "total", Ttot[m].mean().round(2), "up", up[m].mean().round(2), "upq", upq[m].mean().round(2), "tg", tg[m].mean().round(1))
    print("   tg distribution", np.unique(np.round(tg[m]), return_counts=True))
    print("   ctrl", np.bincount(CTRL[m], minlength=3))
# within stay: tabulate total load
print("P(stay | phase 3, total load bin)")
for lo in range(0, 30, 3):
    mm = m3 & (Ttot >= lo) & (Ttot < lo+3)
    if mm.sum(): print(f"  total {lo}-{lo+3}: {stay[mm].sum()}/{mm.sum()}")
print("P(stay | phase 3, D (EW-left own load) value)")
for dv in np.unique(D[m3]):
    mm = m3 & (D == dv)
    print(f"  D={dv:.0f}: {stay[mm].sum()}/{mm.sum()}")
print("P(stay | phase 3, tg)")
for t in np.unique(np.round(tg[m3])):
    mm = m3 & (np.round(tg) == t); print(f"  tg={t}: {stay[mm].sum()}/{mm.sum()}")
# sample of stay states
idx = np.where(stay)[0][:12]
for k in idx:
    print(np.round(X[k, 0:12]*3).astype(int), np.round(X[k, 12:24]*3).astype(int), "up", np.round(X[k, 32:40]*34).astype(int), "tg", round(tg[k]), "logits", LG[k].round(1))
