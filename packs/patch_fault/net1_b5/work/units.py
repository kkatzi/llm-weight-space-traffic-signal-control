"""Activation statistics and ablation importance of hidden units (v1 weights, recorded states with E2 substitution)."""
import sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import tools
np.set_printoptions(precision=3, suppress=True, linewidth=220)
S = tools.states(); X = S["obs"].astype(float); c = S["controller"]
w = tools.load(os.path.join(HERE, "work", "v1.npz"))
# use non-gridlock states: fixed + actuated + start-network episodes 0,1
m = (c > 0) | (S["episode"] < 2)
X = X[m]; ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
a1 = X @ w["W1"].T + w["b1"]
base = tools.greedy(w, X)
def logits_from_h1(H1):
    return np.tanh(H1 @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]
def logits_from_h2(H2):
    return H2 @ w["W3"].T + w["b3"]
r1 = []
for j in range(64):
    H = h1.copy(); H[:, j] = h1[:, j].mean()
    r1.append(((logits_from_h1(H).argmax(1) != base)[ok]).mean())
r2 = []
for k in range(64):
    H = h2.copy(); H[:, k] = h2[:, k].mean()
    r2.append(((logits_from_h2(H).argmax(1) != base)[ok]).mean())
r1 = np.array(r1); r2 = np.array(r2)
o1 = np.argsort(r1)[:12]; o2 = np.argsort(r2)[:12]
print("h1 least important:")
for j in o1:
    print(f"  j={j:2d} flip={r1[j]:.4f} mean={h1[:, j].mean():+.3f} std={h1[:, j].std():.3f} min={h1[:, j].min():+.3f} max={h1[:, j].max():+.3f} pre[min,max]=[{a1[:, j].min():+.2f},{a1[:, j].max():+.2f}]")
print("h2 least important:")
for k in o2:
    print(f"  k={k:2d} flip={r2[k]:.4f} mean={h2[:, k].mean():+.3f} std={h2[:, k].std():.3f} min={h2[:, k].min():+.3f} max={h2[:, k].max():+.3f} W3[:,k]={w['W3'][:, k]} sum|W2[k,:]|={np.abs(w['W2'][k]).sum():.2f}")
print(tools.trainability(w))
