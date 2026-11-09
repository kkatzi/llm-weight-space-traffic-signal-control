import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; ctrl = S["controller"]
names = tools.names()
L = tools.forward(w, X)
net = L.argmax(1)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
for p in range(4):
    m = ok & (cur==p)
    print("cur",p,"mean logits", L[m].mean(0), " t_green q:", np.quantile(X[m,45],[0,.5,.9,1]))
# feature means in ok states
print("feature means ok:")
for i in range(40): print(f"{names[i]:6s} mean {X[ok,i].mean():.3f} q90 {np.quantile(X[ok,i],.9):.3f} max {X[ok,i].max():.3f}")
# phase durations in network episodes
m0 = ctrl==0
print("t_green when net switches (ctrl0, ok):", np.unique(np.round(X[m0&ok&(net!=cur),45]*60), return_counts=True))
print("t_green when net stays (ctrl0, ok):", np.unique(np.round(X[m0&ok&(net==cur),45]*60), return_counts=True))
print("t_green all ok:", np.unique(np.round(X[ok,45]*60), return_counts=True))
