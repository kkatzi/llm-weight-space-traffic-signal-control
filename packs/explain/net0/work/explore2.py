import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
names = tools.names()
# t_green distribution at ok states per current phase, network's own run
for p in range(4):
    m = ok & (cur==p) & (ctrl==0)
    tg = np.round(X[m,45]*60).astype(int)
    vals, cnt = np.unique(tg, return_counts=True)
    keep = [ (net[m][tg==v]==p).mean() for v in vals]
    print("phase",p, list(zip(vals,cnt,np.round(keep,2))))
# all-controller t_green
print("weights of ok and t_green, phase onehots on W1: norms")
W1=w["W1"]
print("col norms", np.round(np.linalg.norm(W1,axis=0),2))
