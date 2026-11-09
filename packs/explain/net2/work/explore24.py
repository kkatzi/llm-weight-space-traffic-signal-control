from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
def grads(Xs):
    a1 = Xs @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
    return np.einsum('ok,nk,kj,nj,ji->noi', w["W3"], 1-h2**2, w["W2"], 1-h1**2, w["W1"])
srt = np.argsort(-LG, 1); gap = LG[np.arange(len(X)), srt[:, 0]] - LG[np.arange(len(X)), srt[:, 1]]
r = np.zeros(len(X), int)
for sel_name, selfn in [("all ok", lambda p: OK & (PH == p)), ("near boundary", lambda p: OK & (PH == p) & (gap < 3))]:
    for p in range(4):
        m = selfn(p)
        G = grads(X[m]).mean(0); xbar = X[m].mean(0); Lbar = LG[m].mean(0)
        mm = OK & (PH == p)
        Lin = Lbar + (X[mm] - xbar) @ G.T
        Lin[:, p] = -99
        r[mm] = Lin.argmax(1)
    report(r, label=f"Taylor ({sel_name}) per phase")
