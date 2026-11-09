import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); c = S["controller"]
nm = tools.names()
W1 = w["W1"]
print("column norms of W1:")
for i, n in enumerate(nm):
    print(f"{i:2d} {n:8s} {np.linalg.norm(W1[:, i]):.3f}")
pairs = [(2, 5), (2, 11), (5, 11), (8, 11), (8, 2), (14, 17), (17, 23), (14, 23), (20, 23), (0, 9), (3, 9), (12, 21), (15, 21), (4, 10), (16, 22)]
print("cosine between columns")
for a, b in pairs:
    ca, cb = W1[:, a], W1[:, b]
    print(nm[a], nm[b], round(float(ca @ cb / np.linalg.norm(ca) / np.linalg.norm(cb)), 3))
# gradient of logits wrt inputs at the recorded states
def grad(w, X):
    a1 = X @ w["W1"].T + w["b1"]; h1 = np.tanh(a1)
    a2 = h1 @ w["W2"].T + w["b2"]; h2 = np.tanh(a2)
    # dlogit_k/dx = W3[k] * (1-h2^2) @ W2 * (1-h1^2) @ W1
    G = np.einsum("kj,nj,ji,ni,im->nkm", w["W3"], 1 - h2**2, w["W2"], 1 - h1**2, w["W1"])
    return G
G = grad(w, X)
Gm = G.mean(0)
print("mean dlogit/dx  (rows phases 0..3), selected inputs")
for i in [2, 5, 8, 11, 14, 17, 20, 23, 0, 3, 6, 9, 1, 4, 7, 10, 25, 27, 29, 31, 33, 35, 37, 39]:
    print(f"{nm[i]:7s}", Gm[:, i], " centered (k - mean):", Gm[:, i] - Gm[:, i].mean())
