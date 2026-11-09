from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
Gs = np.mean([GP[p] for p in range(4)], 0)
# coefficient per unit for each raw feature, centered; stop-line features combined into v = 3(q+d), upstream as d*34.13 vehicles
c = (Gs[:, 0:12] + Gs[:, 12:24]) / 2 / 3
cu = Gs[:, 32:40] / 34.13
c_all = np.concatenate([c, cu], 1); c_all = c_all - c_all.mean(0, keepdims=True)
F = np.concatenate([np.stack([v(X, n) for n in LANES], 1), X[:, 32:40]*34.13], 1)  # 20 features
idx = {n: i for i, n in enumerate(LANES + ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u'])}
def run(groups, label, show=False):
    Gl = np.stack([sum(wt*F[:, idx[l]] for l, wt in g.items()) for g in groups], 1)
    M = np.array([[np.mean([c_all[o, idx[l]]/wt for l, wt in g.items()]) for g in groups] for o in range(4)])
    r = np.zeros(len(X), int); bs = {}
    for p in range(4):
        m_ref = OK & (PH == p) & (gap < 3)
        b = LG[m_ref].mean(0) - Gl[m_ref].mean(0) @ M.T
        mm = OK & (PH == p)
        L = Gl[mm] @ M.T + b; L[:, p] = -99; r[mm] = L.argmax(1); bs[p] = b
    report(r, label=label)
    if show:
        print(M.round(3))
        for p in range(4): print(" b", p, bs[p].round(2))
    return M, bs, r
A = {'N0': .25, 'N1': 1, 'S0': .75, 'S1': .75}; B = {'N2': 1, 'S2': 1}
Cs = {'E1': 1, 'W1': 1.5}; Cr = {'E0': 1, 'W0': 1}; D = {'E2': 1, 'W2': 1}
UN = {'N1u': 1, 'S1u': 1}; UE = {'E0u': 1, 'E1u': 1, 'W0u': 1, 'W1u': 1}; UN0 = {'N0u': 1, 'S0u': 1}
run([A, B, Cs, Cr, D], "5 groups")
run([A, B, Cs, Cr, D, UN, UE], "5 groups + 2 upstream", show=True)
run([A, B, Cs, Cr, D, UN, UE, UN0], "5 groups + 3 upstream")
