import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
G = {0:[0,1,6,7], 1:[2,8], 2:[3,4,9,10], 3:[5,11]}   # stop-line lanes served per phase
def served(p): return G[p]
for ci, cname in enumerate(S["controller_names"]):
    m = c==ci
    Xc, Ac, tc, epc = X[m], A[m], t[m], ep[m]
    ph = Xc[:,40:44].argmax(1); ok = Xc[:,44]>0.5
    sw = ok & (Ac!=ph)
    print("=====", cname, "n", m.sum(), "episodes end:", [int(tc[epc==e].max()) for e in np.unique(epc)])
    print(" green length at switch (s) pct 10/50/90:", np.percentile(Xc[sw,45]*60,[10,50,90]), " per phase median:", [float(np.median(Xc[sw&(ph==p),45]*60)) for p in range(4)])
    print(" phase share of time:", np.bincount(ph,minlength=4)/len(ph))
    # stuck analysis: ok, stay, served lanes empty (density 0), other lanes with queue
    for p in range(4):
        srv = served(p); oth = [i for i in range(12) if i not in srv]
        stay = ok & (Ac==p) & (ph==p)
        empty = Xc[:,12+np.array(srv)].max(1) < 0.01
        othq = Xc[:,np.array(oth)].max(1) > 0.01
        cnt = (stay & empty & othq).sum()
        print(f"  phase {p}: stay-decisions {stay.sum()}, of which served lanes empty & other queue>0: {cnt}; mean served queue at switch {Xc[sw&(ph==p)][:,srv].sum(1).mean() if (sw&(ph==p)).any() else 0:.2f}")
