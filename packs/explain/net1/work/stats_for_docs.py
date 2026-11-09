import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
S = tools.states(); X = S["obs"]; ctrl = S["controller"]; w = tools.load("start_weights.npz"); net = tools.greedy(w, X)
ok = X[:,44] > 0.5; cur = X[:,40:44].argmax(1)
m0 = ok & (ctrl==0)
print("network's own runs: honoured decisions", m0.sum())
T = np.zeros((4,4), int)
for a,b in zip(cur[m0], net[m0]): T[a,b]+=1
print(T); print((T/T.sum(1,keepdims=True)).round(3))
print("share of honoured decisions that extend:", (net[m0]==cur[m0]).mean().round(3))
# green durations in network runs: t_green at switch
sw = m0 & (net!=cur)
print("green age at switch (s):", dict(zip(*np.unique(np.round(X[sw,45]*60).astype(int), return_counts=True))))
# share of time each phase is green in network runs (all decisions)
a0 = ctrl==0
print("share of decisions with each phase green (network runs):", np.bincount(cur[a0], minlength=4)/a0.sum())
