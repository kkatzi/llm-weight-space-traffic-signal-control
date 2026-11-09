from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
srt = np.argsort(-LG, 1); gap = LG[np.arange(len(X)), srt[:, 0]] - LG[np.arange(len(X)), srt[:, 1]]
V = {n: v(X, n) for n in LANES}
A = 0.25*V['N0'] + V['N1'] + 0.75*V['S0'] + 0.75*V['S1']
B = V['N2'] + V['S2']
C = 0.75*V['E0'] + V['E1'] + 0.75*V['W0'] + 1.5*V['W1']
D = V['E2'] + V['W2']
Gl = np.stack([A, B, C, D], 1)
M = np.array([[1.0, 0, 0, 0.2],
              [0, 1.35, 0, 0],
              [0.5, -0.55, 1.25, 0],
              [-0.25, -0.15, 0.45, 1.55]])
def run(M, allow_stay=False, label=""):
    r = np.zeros(len(X), int); bs = {}
    for p in range(4):
        m_ref = OK & (PH == p) & (gap < 3)
        b = (LG[m_ref].mean(0) - Gl[m_ref].mean(0) @ M.T * 1.13)
        b = b / 1.13
        mm = OK & (PH == p)
        L = Gl[mm] @ M.T + b
        if not allow_stay: L[:, p] = -99
        r[mm] = L.argmax(1); bs[p] = b
    report(r, label=label)
    return r, bs
r, bs = run(M, label="rounded group model")
for p in range(4): print(p, (bs[p] - bs[p][[j for j in range(4) if j != p]].mean()).round(2))
r2, _ = run(M, allow_stay=True, label="rounded group model, stay allowed")
for p in range(4): print(p); print(confusion(r, OK & (PH == p)))
