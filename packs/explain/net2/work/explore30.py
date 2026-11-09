from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
GR = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
OWN = {l: g for g, ls in GR.items() for l in ls}
Gs = np.mean([GP[p] for p in range(4)], 0)
c = (Gs[:, 0:12] + Gs[:, 12:24]) / 2 / 3; c = c - c.mean(0, keepdims=True)  # per unit v
V = np.stack([v(X, n) for n in LANES], 1)  # lane loads
def run(wl, label, per_phase_G=False):
    # group loads with lane weights wl; M[o,g] = mean over lanes of c[o,l]/wl[l]
    Gload = np.stack([sum(wl[l]*V[:, l] for l in GR[g]) for g in range(4)], 1)
    r = np.zeros(len(X), int)
    Ms = {}
    for p in range(4):
        if per_phase_G:
            G = GP[p]; cp = (G[:, 0:12] + G[:, 12:24]) / 2 / 3; cp = cp - cp.mean(0, keepdims=True)
        else: cp = c
        M = np.array([[np.mean([cp[o, l]/wl[l] for l in GR[g]]) for g in range(4)] for o in range(4)])
        m_ref = OK & (PH == p) & (gap < 3)
        b = LG[m_ref].mean(0) - Gload[m_ref].mean(0) @ M.T
        mm = OK & (PH == p)
        L = Gload[mm] @ M.T + b
        L[:, p] = -99; r[mm] = L.argmax(1)
        Ms[p] = (M, b)
    report(r, label=label)
    return Ms
ones = {l: 1.0 for l in range(12)}
run(ones, "group sums unweighted, shared M")
ownw = {}
for l in range(12):
    g = OWN[l]; ownw[l] = c[g, l] - np.mean([c[o, l] for o in range(4) if o != g])
print({LANES[l]: round(ownw[l], 2) for l in range(12)})
Ms = run(ownw, "group sums own-weighted, shared M")
run(ownw, "group sums own-weighted, per-phase M", per_phase_G=True)
rw = {0: 0.25, 1: 1, 2: 1, 3: 0.75, 4: 1, 5: 1, 6: 0.75, 7: 0.75, 8: 1, 9: 0.75, 10: 1.5, 11: 1}
Ms2 = run(rw, "group sums rounded weights, shared M")
for p in range(4):
    M, b = Ms2[p]
    print("phase", p); print(M.round(2)); print("b", b.round(2))
