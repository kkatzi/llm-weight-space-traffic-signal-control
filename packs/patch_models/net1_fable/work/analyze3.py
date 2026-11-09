import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
X, c = S["obs"], S["controller"]
ok = X[:,44]>0.5
Xo = X[ok]            # honoured states from all controllers (more diverse)
g0 = tools.greedy(w, Xo)
h1, h2 = tools.hidden(w, Xo)
ph = Xo[:,40:44].argmax(1)
def r2_phase(H):
    out=[]
    for k in range(H.shape[1]):
        v = H[:,k]; tot = v.var()
        within = sum(((v[ph==p]-v[ph==p].mean())**2).sum() for p in range(4))/len(v)
        out.append(1-within/tot if tot>0 else 0)
    return np.array(out)
r1 = r2_phase(h1); r2 = r2_phase(h2)
print("layer1 units with phase R2 > 0.5:", [(k, round(r1[k],2)) for k in np.argsort(-r1)[:10]])
print("layer2 units with phase R2 > 0.5:", [(k, round(r2[k],2)) for k in np.argsort(-r2)[:10]])
# severing cost layer1: zero W2[:,k]
flips1=[]
for k in range(64):
    w2 = {kk:v.copy() for kk,v in w.items()}; w2["W2"][:,k]=0
    flips1.append((tools.greedy(w2,Xo)!=g0).mean())
flips1=np.array(flips1)
print("layer1 severing flip rate, 10 cheapest:", [(k, round(flips1[k],3)) for k in np.argsort(flips1)[:10]])
# severing + mean compensation into b2
flips1c=[]
for k in range(64):
    w2 = {kk:v.copy() for kk,v in w.items()}; mu=h1[:,k].mean(); w2["b2"]=w2["b2"]+w2["W2"][:,k]*mu; w2["W2"][:,k]=0
    flips1c.append((tools.greedy(w2,Xo)!=g0).mean())
flips1c=np.array(flips1c)
print("layer1 severing+compensation flip rate, 10 cheapest:", [(k, round(flips1c[k],3)) for k in np.argsort(flips1c)[:10]])
# layer2 removal cost: zero W3[:,j]
flips2=[]; flips2c=[]
for j in range(64):
    w2 = {kk:v.copy() for kk,v in w.items()}; w2["W3"][:,j]=0
    flips2.append((tools.greedy(w2,Xo)!=g0).mean())
    w2 = {kk:v.copy() for kk,v in w.items()}; mu=h2[:,j].mean(); w2["b3"]=w2["b3"]+w2["W3"][:,j]*mu; w2["W3"][:,j]=0
    flips2c.append((tools.greedy(w2,Xo)!=g0).mean())
flips2=np.array(flips2); flips2c=np.array(flips2c)
print("layer2 removal flip rate, 12 cheapest:", [(j, round(flips2[j],3)) for j in np.argsort(flips2)[:12]])
print("layer2 removal+comp flip rate, 12 cheapest:", [(j, round(flips2c[j],3)) for j in np.argsort(flips2c)[:12]])
print("h1 std per unit (10 smallest):", [(k, round(h1[:,k].std(),3)) for k in np.argsort(h1.std(0))[:10]])
print("h2 std per unit (10 smallest):", [(k, round(h2[:,k].std(),3)) for k in np.argsort(h2.std(0))[:10]])
np.savez("work/unit_stats.npz", r1=r1, r2=r2, flips1=flips1, flips1c=flips1c, flips2=flips2, flips2c=flips2c)
