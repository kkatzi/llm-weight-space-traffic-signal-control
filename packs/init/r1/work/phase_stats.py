"""Per-phase vehicle-count statistics on the recorded states (descriptive only)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states()
X = S["obs"].astype(np.float64)
A = S["act"]; C = S["controller"]; E = S["episode"]; T = S["time_s"]
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
np.set_printoptions(precision=2, suppress=True, linewidth=160)

CAP_S, CAP_U = 22.4 / 7.5, 256 / 7.5
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5

def lanes(p):
    return {0: ["N0", "N1", "S0", "S1"], 1: ["N2", "S2"], 2: ["E0", "E1", "W0", "W1"], 3: ["E2", "W2"]}[p]

def ups(p):  # (lane, share)
    return {0: [("N0u", 1), ("S0u", 1), ("N1u", .5), ("S1u", .5)], 1: [("N1u", .5), ("S1u", .5)],
            2: [("E0u", 1), ("W0u", 1), ("E1u", .5), ("W1u", .5)], 3: [("E1u", .5), ("W1u", .5)]}[p]

H = np.zeros((len(X), 4)); N = np.zeros((len(X), 4)); UH = np.zeros((len(X), 4)); UN = np.zeros((len(X), 4))
for p in range(4):
    for l in lanes(p):
        H[:, p] += CAP_S * X[:, ix["q_" + l]]; N[:, p] += CAP_S * X[:, ix["d_" + l]]
    for l, s in ups(p):
        UH[:, p] += s * CAP_U * X[:, ix["q_" + l]]; UN[:, p] += s * CAP_U * X[:, ix["d_" + l]]

for c in range(2):
    m = (C == c) & ok
    print("controller", S["controller_names"][c], "ok states", m.sum())
    cur = ph[m]
    r = np.arange(m.sum())
    for nm, M in [("H", H), ("N", N), ("UH", UH), ("UN", UN)]:
        Mm = M[m]
        print(f"  {nm}: current phase mean {Mm[r, cur].mean():.2f}; others mean {(Mm.sum(1) - Mm[r, cur]).mean() / 3:.2f}")
        for p in range(4):
            sel = cur == p
            print(f"     cur={p}: own {Mm[sel, p].mean():.2f}  per-phase mean {Mm[sel].mean(0)}")
    # action stats
    keep = A[m] == cur
    print("  keep rate", keep.mean(), "per cur phase", [round(float(keep[cur == p].mean()), 2) for p in range(4)])
    # chosen phase vs argmax H among others
    Hm = H[m].copy(); Hm[r, cur] = -1
    sw = ~keep
    print("  when switching, chosen == argmax(H of others):", (A[m][sw] == Hm[sw].argmax(1)).mean())
    Wm = (H + 0.5 * (N - H) + UH + 0.3 * (UN - UH))[m].copy(); Wm[r, cur] = -1
    print("  when switching, chosen == argmax(W of others):", (A[m][sw] == Wm[sw].argmax(1)).mean())

# time profile of total vehicles
for c in range(2):
    m = C == c
    tot = (N + UN)[m].sum(1) / 1.0
    for e in np.unique(E[m])[:2]:
        mm = E[m] == e
        t = T[m][mm]
        print("ctrl", c, "ep", e, "total veh (approx) by 5 min:",
              [round(float(tot[mm][(t >= a) & (t < a + 300)].mean()), 1) for a in range(0, 3900, 300)])
