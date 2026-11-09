import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; ctrl = S["controller"]
names = tools.names(); ix = {n:i for i,n in enumerate(names)}
L = tools.forward(w, X); net = L.argmax(1)
ok = X[:,44] > 0.5; cur = X[:,40:44].argmax(1)
def report(decide, label=""):
    r = np.array([decide(x) for x in X])
    agree = (r==net)
    print(f"{label}: fidelity ok={agree[ok].mean():.4f}  per ctrl:", [round(agree[ok&(ctrl==c)].mean(),3) for c in range(3)],
          " per cur:", [round(agree[ok&(cur==p)].mean(),3) for p in range(4)])
    return r
def confusion(r, p):
    m = ok & (cur==p)
    T = np.zeros((4,4),int)
    for a,b in zip(net[m], r[m]): T[a,b]+=1
    print(f"cur {p}: rows=net, cols=rules\n{T}")
