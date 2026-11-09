from harness import *
GG = np.load("GG.npy")
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
B = {}
def mkdec(Gfun, label, nostay=False):
    Bs = {}
    for p in range(4):
        m = ok & (cur==p) & (gap<2.0)
        G = Gfun(p)
        Bs[p] = (L[m] - X[m] @ G.T).mean(0)
    def dec(x):
        p = int(np.argmax(x[40:44])); l = Bs[p] + Gfun(p) @ x
        if nostay: l[p] = -1e9
        return int(np.argmax(l))
    return report(dec, label)
mkdec(lambda p: GG[p], "full per-cur linear")
mkdec(lambda p: GG[p], "full per-cur linear, no stay", nostay=True)
def noup(p):
    G = GG[p].copy(); G[:,24:40] = 0; return G
mkdec(noup, "per-cur, upstream removed")
def noupq(p):
    G = GG[p].copy(); G[:,24:32] = 0; return G
mkdec(noupq, "per-cur, upstream queues removed (keep up density)")
def merged_qd(p):
    G = GG[p].copy()
    a = (G[:,0:12]+G[:,12:24])/2; G[:,0:12]=a; G[:,12:24]=a; return G
mkdec(merged_qd, "per-cur, q and d same weight")
def donly(p):
    G = GG[p].copy(); G[:,12:24] += G[:,0:12]*0.5; G[:,0:12]=0; return G
mkdec(donly, "per-cur, q folded into d (x0.5)")
Gavg = np.mean([GG[p]-GG[p].mean(0) for p in range(4)],0)
mkdec(lambda p: Gavg, "shared gradient across curs")
