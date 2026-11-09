import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
S = tools.states(); X = S["obs"]; A = S["act"]
print("queue quantiles all lanes", np.round(np.quantile(X[:,:12],[.5,.9,.99]),3))
print("phase onehot sums", X[:,40:44].sum(0), "unique rowsums", np.unique(X[:,40:44].sum(1)))
cur = X[:,40:44].argmax(1)
print("ok share by phase", [round(X[cur==p,44].mean(),3) for p in range(4)])
# mean queue of each lane when its phase is current
idx = {'N0':0,'N1':1,'N2':2,'E0':3,'E1':4,'E2':5,'S0':6,'S1':7,'S2':8,'W0':9,'W1':10,'W2':11}
for p in range(4): print("phase",p,"mean q", np.round(X[cur==p,:12].mean(0),2))
print("state names sample", tools.block(X[0]))
