import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=150)
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
print("controllers", S.get("controller_names"))
print(X.shape, np.bincount(c))
names = tools.names()
g = tools.greedy(w, X)
print("agreement greedy vs recorded act per controller:", [float((g[c==i]==A[c==i]).mean()) for i in range(3)])
# network states only
m = c==0
Xn, gn = X[m], g[m]
ph = Xn[:,40:44].argmax(1)
ok = Xn[:,44]>0.5
print("phase distribution of greedy decisions (net states):", np.bincount(gn, minlength=4))
print("current phase distribution:", np.bincount(ph, minlength=4))
# transition matrix when ok
T = np.zeros((4,4),int)
for p,a in zip(ph[ok], gn[ok]): T[p,a]+=1
print("transitions when ok (row=current, col=chosen):\n", T)
# green durations: t_green at switch
print("t_green*60 when switching (ok):", np.percentile(Xn[ok & (gn!=ph),45]*60, [10,25,50,75,90]))
print(tools.trainability(w))
# per-episode summary of network states
for e in np.unique(ep[m]):
    mm = m & (ep==e)
    print("ep", e, "n", mm.sum(), "max time", t[mm].max(), "mean total queue", X[mm,:12].sum(1).mean().round(2), "mean up queue", X[mm,24:32].sum(1).mean().round(2))
