import sys; sys.path.insert(0, '.')
import tools, numpy as np
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(np.float64); C = S['controller']
names = tools.names()
print("W1 col norms for 11, 23 vs others:")
cn = np.linalg.norm(w['W1'], axis=0)
for i in np.argsort(-cn): print(f"{names[i]:8s} {cn[i]:.2f}", end=" | ")
print()
a0 = tools.greedy(w, X)
print("act dist start-net on all states", np.bincount(a0, minlength=4))
# proxy imputation
Xp = X.copy(); Xp[:,11] = X[:,5]; Xp[:,23] = X[:,17]
ap = tools.greedy(w, Xp)
ok = X[:,44] > 0.5
print("proxy act dist", np.bincount(ap, minlength=4))
print("agreement all", (a0==ap).mean(), "honoured", (a0[ok]==ap[ok]).mean())
for c in range(3):
    m = (C==c)&ok
    print(c, "start", np.bincount(a0[m],minlength=4), "proxy", np.bincount(ap[m],minlength=4))
# transitions in decisions
M = np.zeros((4,4),int)
for i,j in zip(a0[ok],ap[ok]): M[i,j]+=1
print(M)
# sensitivity: gradient of logits wrt x11,x23 averaged
def grad(w, X, i, eps=1e-3):
    Xe = X.copy(); Xe[:,i]+=eps
    return (tools.forward(w,Xe)-tools.forward(w,X))/eps
for i in [5,11,17,23,31,39,2,8,14,20]:
    g = grad(w, Xp, i)
    print(names[i], np.round(g.mean(0),2), "logit3-mean(others)", round(float((g[:,3]-g[:,:3].mean(1)).mean()),2))
