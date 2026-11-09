import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"]; A = S["act"]
print("N", len(X), "act hist", np.bincount(A, minlength=4), "ctrl", np.bincount(S["controller"]))
pred = tools.greedy(w, X)
print("net agreement with recorded act", (pred == A).mean())
ok = X[:,44] > 0.5
print("ok share", ok.mean())
print("net phase hist", np.bincount(pred, minlength=4))
print("max-prob median", np.median(tools.probs(w, X).max(1)))
print(tools.trainability(w))
# queue sums per phase group
names = tools.names()
q = X[:, :12]; d = X[:, 12:24]
print("mean queue per lane", np.round(q.mean(0),3))
print("mean up_queue", np.round(X[:,24:32].mean(0),3))
print("phase share", np.round(X[:,40:44].mean(0),3))
print("t_green quantiles", np.round(np.quantile(X[:,45],[.1,.5,.9]),3))
