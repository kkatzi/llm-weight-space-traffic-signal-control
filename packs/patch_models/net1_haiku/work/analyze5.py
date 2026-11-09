import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"]; A = S["act"]; C = S["controller"]
_, H2 = tools.hidden(w, X)
pred = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
idx = {'N0':0,'N1':1,'N2':2,'E0':3,'E1':4,'E2':5,'S0':6,'S1':7,'S2':8,'W0':9,'W1':10,'W2':11}
groups = {0:['N0','N1','S0','S1'],1:['N2','S2'],2:['E0','E1','W0','W1'],3:['E2','W2']}
d = X[:,12:24]
Dg = np.stack([d[:, [idx[n] for n in groups[p]]].mean(1) for p in range(4)],1)
ind = np.eye(4)[cur]
def corr(a,b):
    a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12))
for p in range(4):
    target = ind[:,p]*Dg[:,p]
    cs = np.array([corr(H2[:,j], target) for j in range(64)])
    ci = np.array([corr(H2[:,j], ind[:,p]) for j in range(64)])
    print("phase",p,"best for ind*dens:", [(int(j), round(cs[j],3)) for j in np.argsort(-np.abs(cs))[:3]],
          "| best for ind:", [(int(j), round(ci[j],3)) for j in np.argsort(-np.abs(ci))[:3]])
