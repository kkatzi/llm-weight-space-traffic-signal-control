from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
Gs = np.mean([GP[p] for p in range(4)], 0)
c = (Gs[:, 0:12] + Gs[:, 12:24]) / 2 / 3; c = c - c.mean(0, keepdims=True)
V = np.stack([v(X, n) for n in LANES], 1)
def run(groups, label, show=False):
    # groups: list of dict lane->weight
    Gl = np.stack([sum(wt*V[:, l] for l, wt in g.items()) for g in groups], 1)
    M = np.array([[np.mean([c[o, l]/wt for l, wt in g.items()]) for g in groups] for o in range(4)])
    r = np.zeros(len(X), int); bs = {}
    for p in range(4):
        m_ref = OK & (PH == p) & (gap < 3)
        b = LG[m_ref].mean(0) - Gl[m_ref].mean(0) @ M.T
        mm = OK & (PH == p)
        L = Gl[mm] @ M.T + b; L[:, p] = -99; r[mm] = L.argmax(1); bs[p] = b
    report(r, label=label)
    if show:
        print(M.round(2))
        for p in range(4): print(" b", p, bs[p].round(2))
    return M, bs
L_ = {n: i for i, n in enumerate(LANES)}
A = {L_['N0']: .25, L_['N1']: 1, L_['S0']: .75, L_['S1']: .75}
B = {L_['N2']: 1, L_['S2']: 1}
C = {L_['E0']: .75, L_['E1']: 1, L_['W0']: .75, L_['W1']: 1.5}
Cs = {L_['E1']: 1, L_['W1']: 1.5}; Cr = {L_['E0']: 1, L_['W0']: 1}
D = {L_['E2']: 1, L_['W2']: 1}
run([A, B, C, D], "4 groups")
run([A, B, Cs, Cr, D], "5 groups (EW straight / EW right split)", show=True)
As = {L_['N1']: 1, L_['S1']: 1}; Ar = {L_['N0']: .3, L_['S0']: 1}
run([As, Ar, B, Cs, Cr, D], "6 groups (NS split too)", show=True)
