import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
names = tools.names()
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
def grads(X):
    a1 = X @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
    # d logits / dx = W3 diag(1-h2^2) W2 diag(1-h1^2) W1
    G = np.einsum('ok,nk,kj,nj,ji->noi', w["W3"], 1-h2**2, w["W2"], 1-h1**2, w["W1"])
    return G
G = grads(X[ok])
print("mean gradient of each logit wrt features, ok states (scaled by 1/3 for stop-line lanes = one vehicle)")
for p in range(4):
    m = ph[ok] == p
    g = G[m].mean(0)
    print(f"--- current phase {p}, n={m.sum()}")
    print("        " + " ".join(f"{n[:6]:>6s}" for n in names[:24]))
    for o in range(4):
        print(f"logit{o} " + " ".join(f"{v/3:6.2f}" for v in g[o, :24]))
    print("        " + " ".join(f"{n[:6]:>6s}" for n in names[24:46]))
    for o in range(4):
        print(f"logit{o} " + " ".join(f"{v/34:6.2f}" if i<16 else f"{v:6.2f}" for i, v in enumerate(g[o, 24:46])))
