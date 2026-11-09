import numpy as np, tools
w0 = tools.load("start_weights.npz"); S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
names = tools.names(); idx={n:i for i,n in enumerate(names)}
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5; tg = np.round(X[:,45]*60).astype(int)
same = (c[1:]==c[:-1])&(ep[1:]==ep[:-1])&(t[1:]-t[:-1]==5)
nxt = np.full(len(X), -1); nxt[:-1][same] = np.where(same)[0]+1
L = {2: ["E0","E1","W0","W1"], 0: ["N0","N1","S0","S1"], 1:["N2","S2"], 3:["E2","W2"]}
for p in (2,0,1,3):
    dsum = sum(X[:, idx["d_"+l]] for l in L[p]); qsum = sum(X[:, idx["q_"+l]] for l in L[p])
    for ci in (0,2):
        m = (c==ci)&ok&(ph==p)&(tg==12)
        stay = A[m]==p
        print(f"ctrl {ci} phase {p} at 12s: n={m.sum()} stays={stay.sum()}  density-sum quantiles {np.round(np.quantile(dsum[m],[.1,.25,.5,.75,.9]),2)}  mean dsum stay/switch {dsum[m][stay].mean() if stay.any() else None:.2f}/{dsum[m][~stay].mean():.2f}")
    # for stays (any controller) with a following state: how much did stop-line density on these lanes drop in the 5 s?
    m = ok&(ph==p)&(A==p)&(nxt>=0)
    m2 = m & (ph[nxt]==p)
    j = nxt[m2]
    d0 = dsum[m2]; d1 = dsum[j]
    print(f"   phase {p} stays (all ctrls, n={m2.sum()}): mean dsum before {d0.mean():.2f} after {d1.mean():.2f}; share with dsum before >=0.67: {(d0>=0.67).mean():.2f}")
    # relation: among stays, density after vs before by bins
    for lo,hi in [(0,0.34),(0.34,0.67),(0.67,1.0),(1.0,1.34),(1.34,9)]:
        s=(d0>=lo)&(d0<hi)
        if s.sum()>5: print(f"      dsum in [{lo},{hi}): n={s.sum()} mean after {d1[s].mean():.2f}")
# At 12 s in phase 2 (net), how does the density sum relate to the net's own logit gap (best other - current)?
m=(c==0)&ok&(ph==2)&(tg==12)
lg=tools.forward(w0,X[m]); gap=np.sort(lg,1)[:,-1]-lg[:,2]; gap[gap==0]=(np.sort(lg,1)[:,-1]-np.sort(lg,1)[:,-2])[gap==0]*-1
dsum = sum(X[m, idx["d_"+l]] for l in L[2])
for lo,hi in [(0,0.34),(0.34,0.67),(0.67,1.0),(1.0,1.34),(1.34,9)]:
    s=(dsum>=lo)&(dsum<hi); print(f"phase2@12s dsum[{lo},{hi}): n={s.sum()} median gap(best other - current) {np.median(gap[s]) if s.any() else None}")
