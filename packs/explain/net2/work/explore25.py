from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
def grads(Xs):
    a1 = Xs @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
    return np.einsum('ok,nk,kj,nj,ji->noi', w["W3"], 1-h2**2, w["W2"], 1-h1**2, w["W1"])
srt = np.argsort(-LG, 1); gap = LG[np.arange(len(X)), srt[:, 0]] - LG[np.arange(len(X)), srt[:, 1]]
GP = {}; XB = {}; LB = {}
for p in range(4):
    m = OK & (PH == p) & (gap < 3)
    GP[p] = grads(X[m]).mean(0); XB[p] = X[m].mean(0); LB[p] = LG[m].mean(0)
def evalmodel(Gfn, label, feats=None):
    r = np.zeros(len(X), int)
    for p in range(4):
        G = Gfn(p).copy()
        if feats is not None:
            mask = np.zeros(46, bool); mask[feats] = True; G[:, ~mask] = 0
        mm = OK & (PH == p)
        Lin = LB[p] + (X[mm] - XB[p]) @ G.T
        Lin[:, p] = -99; r[mm] = Lin.argmax(1)
    report(r, label=label)
    return r
evalmodel(lambda p: GP[p], "per-phase full")
evalmodel(lambda p: GP[p], "per-phase stop-line only", feats=list(range(24)))
evalmodel(lambda p: GP[p], "per-phase stop-line + upstream", feats=list(range(40)))
evalmodel(lambda p: GP[p], "per-phase q only", feats=list(range(12)))
evalmodel(lambda p: GP[p], "per-phase d only", feats=list(range(12, 24)))
Gs = np.mean([GP[p] for p in range(4)], 0)
evalmodel(lambda p: Gs, "shared weights", feats=list(range(24)))
# own-lane only (each logit uses only its own group's lanes)
GR = {0: [0,1,6,7], 1: [2,8], 2: [3,4,9,10], 3: [5,11]}
def own(G):
    G2 = np.zeros_like(G)
    for o in range(4):
        # relative to mean over the logits: logit o's own lanes coefficient = G[o]-mean(G)
        Gc = G - G.mean(0, keepdims=True)
        for l in GR[o]: G2[o, l] = Gc[o, l]; G2[o, 12+l] = Gc[o, 12+l]
    return G2
evalmodel(lambda p: own(GP[p]), "per-phase own-lane only (centered)")
evalmodel(lambda p: own(Gs), "shared own-lane only (centered)")
