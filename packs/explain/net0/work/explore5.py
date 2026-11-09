import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
names=tools.names()
def grads(Xs):
    a1 = Xs@w["W1"].T+w["b1"]; h1=np.tanh(a1)
    a2 = h1@w["W2"].T+w["b2"]; h2=np.tanh(a2)
    # dlogit/dx = W3 diag(1-h2^2) W2 diag(1-h1^2) W1
    G = np.einsum('ok,nk,kj,nj,ji->noi', w["W3"], 1-h2**2, w["W2"], 1-h1**2, w["W1"])
    return G
for p in range(4):
    m = ok&(cur==p)
    G = grads(X[m]).mean(0)  # 4 x 46
    print("=== current phase",p,"n",m.sum(), "net dist",np.bincount(net[m],minlength=4))
    # center logits: relative to mean
    Gc = G - G.mean(0,keepdims=True)
    for i in range(40):
        print(f"{names[i]:7s}", Gc[:,i])
