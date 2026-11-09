import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names(); idx = {n:i for i,n in enumerate(names)}
# lanes per phase: stop-line queue indices and upstream indices
PH = {0: (["N0","N1","S0","S1"], ["N0u","N1u","S0u","S1u"]),
      1: (["N2","S2"], ["N1u","S1u"]),
      2: (["E0","E1","W0","W1"], ["E0u","E1u","W0u","W1u"]),
      3: (["E2","W2"], ["E1u","W1u"])}
def demand(X):
    D = np.zeros((len(X),4))
    for p,(sl,up) in PH.items():
        D[:,p] = sum(X[:,idx["q_"+l]] for l in sl) + sum(X[:,idx["q_"+l]] for l in up)
    return D
def dens(X):
    D = np.zeros((len(X),4))
    for p,(sl,up) in PH.items():
        D[:,p] = sum(X[:,idx["d_"+l]] for l in sl) + sum(X[:,idx["d_"+l]] for l in up)
    return D
g = tools.greedy(w, X)
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5
m = (c==0)&ok
D = demand(X); Dd = dens(X)
ch = g[m]; best = D[m].argmax(1)
print("chosen == max-queue phase:", (ch==best).mean(), " chosen==max-density phase:", (ch==Dd[m].argmax(1)).mean())
# wasted: chosen phase stop-line+upstream density ~ 0 while another phase has demand
chosen_d = Dd[m][np.arange(m.sum()), ch]; maxd = Dd[m].max(1)
print("chosen density quantiles:", np.quantile(chosen_d,[.05,.25,.5,.75,.95]))
print("share chosen density < 0.05 :", (chosen_d<0.05).mean(), " of which max alt density>0.3:", ((chosen_d<0.05)&(maxd>0.3)).mean())
# by time: during hour vs drain
tt = t[m]
for lo,hi in [(0,900),(900,1800),(1800,2700),(2700,3600),(3600,5400)]:
    s = (tt>=lo)&(tt<hi)
    print(f"t {lo}-{hi}: n={s.sum()} switch rate={(ch[s]!=ph[m][s]).mean():.2f} chosen==maxq={(ch[s]==best[s]).mean():.2f} empty-chosen={(chosen_d[s]<0.05).mean():.2f} action dist={np.bincount(ch[s],minlength=4)/max(1,s.sum())}")
# What does the net do with stop-line queue of left pockets etc. Look at pockets: how often are left pockets full (q>=0.9)?
for l in ["N2","S2","E2","W2"]:
    print(l, "pocket q full share (net eps):", (X[c==0, idx["q_"+l]]>=0.9).mean(), " upstream lane1 q>0.3:", (X[c==0, idx["q_"+l[0]+"1u"]]>0.3).mean())
# drain: time of last vehicle: per episode of net
for ci in range(3):
    for e in np.unique(ep[c==ci]):
        mm=(c==ci)&(ep==e)
        print("ctrl",ci,"ep",e,"end t",t[mm].max(), "n",mm.sum())
