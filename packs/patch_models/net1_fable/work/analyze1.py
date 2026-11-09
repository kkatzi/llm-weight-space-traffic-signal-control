import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
print(S.keys(), X.shape, S.get("controller_names"))
names = tools.names()
g = tools.greedy(w, X)
print("agreement with recorded act (controller 0):", (g[c==0]==A[c==0]).mean())
print("action dist by controller:")
for ci in range(3):
    m = c==ci
    print(ci, np.bincount(A[m], minlength=4)/m.sum(), "net greedy on these states:", np.bincount(g[m], minlength=4)/m.sum())
print(tools.trainability(w))
# phase one-hot and ok, t_green
ph = X[:,40:44].argmax(1)
ok = X[:,44]>0.5
m = c==0
print("net states: transitions matrix (current phase -> chosen), only ok states")
M = np.zeros((4,4),int)
for p,a in zip(ph[m&ok], g[m&ok]): M[p,a]+=1
print(M)
print("switch rate when ok:", (ph[m&ok]!=g[m&ok]).mean())
# green durations in net episodes
for e in np.unique(ep[m]):
    mm = m&(ep==e)
    tg = X[mm,45]*60
    print("episode",e,"n",mm.sum(),"t max",t[mm].max(), "mean t_green at switch (ok & g!=ph):", tg[(ok[mm])&(g[mm]!=ph[mm])].mean() if ((ok[mm])&(g[mm]!=ph[mm])).any() else None)
# per phase, mean t_green at the moment of leaving
for p in range(4):
    sel = m&ok&(ph==p)&(g!=ph)
    print("phase",p,"leaves n",sel.sum(),"mean t_green",(X[sel,45]*60).mean(), "stays n",(m&ok&(ph==p)&(g==ph)).sum())
