import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"]; A = S["act"]; C = S["controller"]
pred = tools.greedy(w, X)
for c in range(3):
    m = C == c
    print("controller", c, "net agreement with its own actions", round((pred[m]==A[m]).mean(),3))
ok = X[:,44] > 0.5
# queue-based pressure per phase: halting share on green lanes
q = X[:, :12]; d = X[:, 12:24]
idx = {'N0':0,'N1':1,'N2':2,'E0':3,'E1':4,'E2':5,'S0':6,'S1':7,'S2':8,'W0':9,'W1':10,'W2':11}
groups = {0:['N0','N1','S0','S1'],1:['N2','S2'],2:['E0','E1','W0','W1'],3:['E2','W2']}
P = np.stack([q[:, [idx[n] for n in groups[p]]].sum(1) for p in range(4)],1)
Dn = np.stack([d[:, [idx[n] for n in groups[p]]].sum(1) for p in range(4)],1)
rule = P.argmax(1)
print("net vs max-queue rule (honoured)", round((pred[ok]==rule[ok]).mean(),3))
print("net vs max-density rule (honoured)", round((pred[ok]==Dn.argmax(1)[ok]).mean(),3))
# current phase: does net stay in current green when ok?
cur = X[:,40:44].argmax(1)
print("net keeps current phase (ok states)", round((pred[ok]==cur[ok]).mean(),3))
print("net keeps current phase (all)", round((pred==cur).mean(),3))
# by controller: how often each controller switches among ok states
for c in range(3):
    m=(C==c)&ok
    print("ctrl",c,"switch share when ok", round((A[m]!=cur[m]).mean(),3))
print("net switch share when ok", round((pred[ok]!=cur[ok]).mean(),3))
