import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"]; ctrl = S["controller"]
names = tools.names()
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)

def grad(x):
    a1 = w["W1"]@x + w["b1"]; h1 = np.tanh(a1)
    a2 = w["W2"]@h1 + w["b2"]; h2 = np.tanh(a2)
    # dlogits/dx = W3 diag(1-h2^2) W2 diag(1-h1^2) W1
    return w["W3"] @ ((1-h2**2)[:,None] * w["W2"]) @ ((1-h1**2)[:,None] * w["W1"])

for p in range(4):
    m = ok & (cur==p)
    G = np.mean([grad(x) for x in X[m]], axis=0)  # 4 x 46
    print(f"=== current phase {p}, n={m.sum()}  (mean grad of logits wrt x, cols 0-39)")
    for j in range(4):
        print(f" logit{j}:", " ".join(f"{names[i]}:{G[j,i]:+.1f}" for i in range(40) if abs(G[j,i])>1.0))
