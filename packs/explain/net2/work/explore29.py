from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
np.set_printoptions(linewidth=250, precision=2, suppress=True)
GR = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
OWN = {l: g for g, ls in GR.items() for l in ls}
Gs = np.mean([GP[p] for p in range(4)], 0)
def own_proper(G):
    cc = (G[:, 0:12] + G[:, 12:24]) / 2
    G2 = np.zeros_like(G)
    for l in range(12):
        g = OWN[l]
        val = cc[g, l] - np.mean([cc[o, l] for o in range(4) if o != g])
        G2[g, l] = val; G2[g, 12+l] = val
    return G2
evalmodel(lambda p: own_proper(GP[p]), "per-phase own-only (proper scale), q+d")
evalmodel(lambda p: own_proper(Gs), "shared own-only (proper scale), q+d")
G2 = own_proper(Gs)
print("own weights per unit v (shared):", {LANES[l]: round(G2[OWN[l], l]/3, 2) for l in range(12)})
for p in range(4):
    icpt = LB[p] - XB[p][:24] @ G2[:, :24].T
    print("phase", p, "intercepts at zero load", icpt.round(2))
