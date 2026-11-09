import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; act = S["act"]; ctrl = S["controller"]
print(S.keys() if hasattr(S,'keys') else None)
print({k: v.shape for k, v in S.items()})
print(S["controller_names"] if "controller_names" in S else None)
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
print("n", len(X), "ok", ok.sum())
cur = X[:,40:44].argmax(1)
print("net decision distribution (ok):", np.bincount(net[ok], minlength=4))
print("stay rate (ok):", (net[ok]==cur[ok]).mean())
for c in range(3):
    m = ok & (ctrl==c)
    print("ctrl", c, "n", m.sum(), "stay", (net[m]==cur[m]).mean())
# transition matrix cur -> net when ok
T = np.zeros((4,4), int)
for a,b in zip(cur[ok], net[ok]): T[a,b]+=1
print(T)
# start_network controller: act equals net?
m = ctrl==0
print("act==net for ctrl0", (act[m]==net[m]).mean())
print(tools.trainability(w))
