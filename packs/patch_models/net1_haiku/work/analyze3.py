import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"]; A = S["act"]; C = S["controller"]
pred = tools.greedy(w, X)
h1, h2 = tools.hidden(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
idx = {'N0':0,'N1':1,'N2':2,'E0':3,'E1':4,'E2':5,'S0':6,'S1':7,'S2':8,'W0':9,'W1':10,'W2':11}
groups = {0:['N0','N1','S0','S1'],1:['N2','S2'],2:['E0','E1','W0','W1'],3:['E2','W2']}
q = X[:, :12]; d=X[:,12:24]
G = np.stack([q[:, [idx[n] for n in groups[p]]].mean(1) for p in range(4)],1)  # mean halting per group
Gd = np.stack([d[:, [idx[n] for n in groups[p]]].mean(1) for p in range(4)],1)
curq = G[np.arange(len(X)), cur]
othq = np.where(np.eye(4)[cur]>0, -1, G).max(1)
for lo,hi in [(0,0.1),(0.1,0.2),(0.2,0.35),(0.35,1.01)]:
    m = ok & (curq>=lo) & (curq<hi)
    if m.sum(): print(f"cur-green queue [{lo},{hi}) n={m.sum()} net switch share {np.mean(pred[m]!=cur[m]):.3f}  ctrl0 switch?")
m=(C==0)&ok
for lo,hi in [(0,0.1),(0.1,0.2),(0.2,0.35),(0.35,1.01)]:
    mm = m & (curq>=lo) & (curq<hi)
    if mm.sum(): print(f"ctrl0 cur queue [{lo},{hi}) n={mm.sum()} switch share {np.mean(A[mm]!=cur[mm]):.3f}")
# correlations of h2 units with current-group queue and with phase onehot
Z = h2
def corr(a,b):
    a=a-a.mean(); b=b-b.mean(); return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-12)
cq = np.array([corr(Z[:,j], curq) for j in range(64)])
print("top |corr| h2 with cur-green queue:", [(j, round(cq[j],2)) for j in np.argsort(-np.abs(cq))[:6]])
ph = np.eye(4)[cur]
for p in range(4):
    cp = np.array([corr(Z[:,j], ph[:,p]) for j in range(64)])
    print("phase",p,"top h2 corr", [(j, round(cp[j],2)) for j in np.argsort(-np.abs(cp))[:3]])
