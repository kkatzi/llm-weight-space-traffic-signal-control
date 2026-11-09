from common import *
exec(open('explore25.py').read().split("evalmodel(lambda p: GP[p], \"per-phase full\")")[0])
Gs = np.mean([GP[p] for p in range(4)], 0)
def qd_comb(G):
    G2 = G.copy(); c = (G[:, 0:12] + G[:, 12:24]) / 2
    G2[:, 0:12] = c; G2[:, 12:24] = c
    return G2
evalmodel(lambda p: qd_comb(GP[p]), "per-phase, q+d combined", feats=list(range(24)))
evalmodel(lambda p: qd_comb(Gs), "shared, q+d combined", feats=list(range(24)))
# sparsify shared combined centered: keep entries with |per-halting-vehicle effect| >= thr
Gc = qd_comb(Gs); Gc = Gc - Gc.mean(0, keepdims=True)
for thr in [0.5, 1.0]:
    def sp(p, thr=thr):
        G = Gc.copy(); eff = G[:, 0:12]*2/3
        keep = np.abs(eff) >= thr
        G[:, 0:12] *= keep; G[:, 12:24] *= keep
        return G
    evalmodel(sp, f"shared combined, sparsified |eff|>={thr}", feats=list(range(24)))
    print("  kept per logit:", (np.abs(Gc[:, 0:12]*2/3) >= thr).sum(1))
