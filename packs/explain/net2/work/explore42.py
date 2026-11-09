from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
F = np.concatenate([np.stack([v(X, n) for n in LANES], 1), X[:, 32:40]*34.13], 1)
idx = {n: i for i, n in enumerate(LANES + ['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u'])}
A_ = {'N0': .25, 'N1': 1, 'S0': .75, 'S1': .75}; B_ = {'N2': 1, 'S2': 1}
Cs = {'E1': 1, 'W1': 1.5}; Cr = {'E0': 1, 'W0': 1}; D_ = {'E2': 1, 'W2': 1}
UN = {'N1u': 1, 'S1u': 1}; UE = {'E0u': 1, 'E1u': 1, 'W0u': 1, 'W1u': 1}
groups = [A_, B_, Cs, Cr, D_, UN, UE]
Gl = np.stack([sum(wt*F[:, idx[l]] for l, wt in g.items()) for g in groups], 1)
r = np.zeros(len(X), int)
for p in range(4):
    G = GP[p]; c = (G[:, 0:12] + G[:, 12:24]) / 2 / 3; cu = G[:, 32:40] / 34.13
    c_all = np.concatenate([c, cu], 1); c_all = c_all - c_all.mean(0, keepdims=True)
    M = np.array([[np.mean([c_all[o, idx[l]]/wt for l, wt in g.items()]) for g in groups] for o in range(4)])
    m_ref = OK & (PH == p) & (gap < 3)
    b = LG[m_ref].mean(0) - Gl[m_ref].mean(0) @ M.T
    mm = OK & (PH == p)
    L = Gl[mm] @ M.T + b; L[:, p] = -99; r[mm] = L.argmax(1)
    cands = [j for j in range(4) if j != p]
    # express relative to first candidate's own coefficient
    print("phase", p); print((M/1.13).round(2))
report(r, label="per-phase M, 5 groups + upstream")
