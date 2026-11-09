import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
ok = X[:,44]>0.5
g0 = tools.greedy(w, X)
h1, h2 = tools.hidden(w, X)
# importance of layer-1 units: zero outgoing W2[:,i] (and optionally compensate b2 with mean)
res1=[]
for i in range(64):
    w2 = {k:v.copy() for k,v in w.items()}
    w2["W2"][:,i]=0
    d = (tools.greedy(w2,X)!=g0)
    w3 = {k:v.copy() for k,v in w.items()}
    w3["b2"] += w3["W2"][:,i]*h1[:,i].mean(); w3["W2"][:,i]=0
    d2 = (tools.greedy(w3,X)!=g0)
    res1.append((i, d[ok].mean(), d2[ok].mean(), h1[:,i].std(), np.abs(w["W2"][:,i]).sum()))
res1.sort(key=lambda r:r[1])
print("layer1 units: (unit, flip rate ok no-comp, flip rate with mean comp, std h1, |W2 col|)")
for r in res1[:12]: print(r)
res2=[]
for j in range(64):
    w2 = {k:v.copy() for k,v in w.items()}
    w2["W3"][:,j]=0
    d = (tools.greedy(w2,X)!=g0)
    w3 = {k:v.copy() for k,v in w.items()}
    w3["b3"] += w3["W3"][:,j]*h2[:,j].mean(); w3["W3"][:,j]=0
    d2 = (tools.greedy(w3,X)!=g0)
    res2.append((j, d[ok].mean(), d2[ok].mean(), h2[:,j].std(), np.abs(w["W3"][:,j]).sum()))
res2.sort(key=lambda r:r[1])
print("layer2 units:")
for r in res2[:12]: print(r)
# stuck decisions per episode
m = c==0
ph = X[:,40:44].argmax(1)
srv=[3,4,9,10]; oth=[0,1,2,5,6,7,8,11]
cond = ok & (ph==2) & (X[:,12+np.array(srv)].max(1)<0.01) & (X[:,np.array(oth)].sum(1)>0.2)
print("condition count per net episode:", [int((cond&m&(ep==e)).sum()) for e in range(6)], "net chooses 2 when cond:", int((cond&m&(g0==2)).sum()))
print("condition count for fixed/actuated states:", int((cond&(c==1)).sum()), int((cond&(c==2)).sum()), " net greedy picks 2 there:", int((cond&(c>0)&(g0==2)).sum()))
# what does net pick when cond holds and it does not pick 2
print("picks when cond & not 2:", np.bincount(g0[cond&(g0!=2)],minlength=4))
