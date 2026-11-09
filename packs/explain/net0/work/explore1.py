import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; act = S["act"]; ctrl = S["controller"]
print(S.keys(), X.shape, S.get("controller_names"))
for k in S: print(k, S[k].shape, S[k].dtype)
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
print("n ok", ok.sum())
cur = X[:,40:44].argmax(1)
print("controller dist", np.bincount(ctrl))
for c in range(3):
    m = (ctrl==c)&ok
    print("ctrl",c,"ok states",m.sum(),"net keep frac", (net[m]==cur[m]).mean(), "net action dist", np.bincount(net[m],minlength=4))
    # transition table cur->net
    T = np.zeros((4,4),int)
    for a,b in zip(cur[m],net[m]): T[a,b]+=1
    print(T)
print(tools.trainability(w))
