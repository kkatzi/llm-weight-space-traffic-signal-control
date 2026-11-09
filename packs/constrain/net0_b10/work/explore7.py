import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]
w = tools.load("start_weights.npz")
L = tools.forward(w, X)
ss = rs[:, 0:4]; legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
ok = X[:, 44] > 0.5
dec = ok & legal.all(1)
CAP_S, CAP_U = 22.4 / 7.5, 256 / 7.5
LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UPX = {0: [24, 28], 2: [26, 30]}       # exclusive upstream lanes (lane 0u)
UPS = {0: [25, 29], 2: [27, 31], 1: [25, 29], 3: [27, 31]}  # shared upstream lanes
def veh(x, p):
    """estimated vehicles present (density) for phase p: stop-line + upstream (exclusive full, shared ~ share)"""
    s = sum(x[12 + l] for l in LANES[p]) * CAP_S
    up = 0.0
    if p in (0, 2):
        up += sum(x[8 + u] for u in UPX[p]) * CAP_U
        up += sum(x[8 + u] for u in UPS[p]) * CAP_U * (55 / 85)
    else:
        up += sum(x[8 + u] for u in UPS[p]) * CAP_U * (30 / 85)
    return s, up
def hal(x, p):
    s = sum(x[l] for l in LANES[p]) * CAP_S
    up = 0.0
    if p in (0, 2):
        up += sum(x[u] for u in UPX[p]) * CAP_U + sum(x[u] for u in UPS[p]) * CAP_U * (55 / 85)
    else:
        up += sum(x[u] for u in UPS[p]) * CAP_U * (30 / 85)
    return s, up
thr = np.where(dec & np.isin(act, [0, 2]) & (act != g))[0]
V = np.array([veh(X[i], act[i]) for i in thr]); H = np.array([hal(X[i], act[i]) for i in thr])
print("through choice: veh stopline q", np.quantile(V[:, 0], [.1, .5, .9]), " upstream", np.quantile(V[:, 1], [.1, .5, .9]))
print("through choice: halt stopline q", np.quantile(H[:, 0], [.1, .5, .9]), " upstream", np.quantile(H[:, 1], [.1, .5, .9]))
# At tg=12 of through phases: remaining vehicles incl upstream
m = np.where(ok & (tg == 12) & np.isin(g, [0, 2]))[0]
R = np.array([veh(X[i], g[i]) for i in m])
print("through at tg=12 remaining veh: stopline", np.quantile(R[:, 0], [.5, .75, .9]), " upstream", np.quantile(R[:, 1], [.5, .75, .9]))
m17 = np.where(ok & (tg == 17) & np.isin(g, [0, 2]))[0]
R = np.array([veh(X[i], g[i]) for i in m17])
print("through at tg=17 remaining veh: stopline", np.quantile(R[:, 0], [.5, .75, .9]), " upstream", np.quantile(R[:, 1], [.5, .75, .9]))
m12 = np.where(ok & (tg == 12) & np.isin(g, [1, 3]))[0]
R = np.array([veh(X[i], g[i]) for i in m12])
print("left at tg=12 remaining veh: stopline", np.quantile(R[:, 0], [.5, .75, .9]), " upstream", np.quantile(R[:, 1], [.5, .75, .9]))
srt = np.argsort(-L, axis=1)
for D in [0.5, 1.0, 1.5, 2.0, 3.0]:
    Ladj = L.copy(); Ladj[:, [0, 2]] -= D
    # hold of current phase not penalized
    for i in range(N):
        if g[i] in (0, 2): Ladj[i, g[i]] += D
    ch = Ladj.argmax(1)
    f = dec & (ch != act)
    print(f"D={D}: changed real decisions {f.sum()}/{dec.sum()}  new choice dist {np.bincount(ch[f], minlength=4)}  from {np.bincount(act[f], minlength=4)}")
