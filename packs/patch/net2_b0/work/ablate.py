import numpy as np, sys
sys.path.insert(0, '.')
import tools
np.set_printoptions(linewidth=220, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); ctl = S["controller"]
ok = X[:,44]>0.5
h1, h2 = tools.hidden(w, X)
a0 = tools.greedy(w, X)
mu1 = h1[ctl==0].mean(0); mu2 = h2[ctl==0].mean(0)
res1=[]
for k in range(64):
    H1 = h1.copy(); H1[:,k] = mu1[k]
    H2 = np.tanh(H1 @ w["W2"].T + w["b2"])
    a = (H2 @ w["W3"].T + w["b3"]).argmax(1)
    res1.append(((a!=a0)&ok&(ctl==0)).sum())
res2=[]
for j in range(64):
    H2 = h2.copy(); H2[:,j] = mu2[j]
    a = (H2 @ w["W3"].T + w["b3"]).argmax(1)
    res2.append(((a!=a0)&ok&(ctl==0)).sum())
res1=np.array(res1); res2=np.array(res2)
print("n honoured net decisions", (ok&(ctl==0)).sum())
o1=np.argsort(res1); o2=np.argsort(res2)
print("L1 least influential:", [(k,res1[k], round(h1[ctl==0,k].std(),3)) for k in o1[:12]])
print("L2 least influential:", [(j,res2[j], round(h2[ctl==0,j].std(),3)) for j in o2[:12]])
np.savez("work/ablation.npz", res1=res1, res2=res2)
