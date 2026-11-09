"""Layer-1 unit survey: activation spread, importance (decision flips when the unit is frozen at its mean) and the
direction its output pushes the logits (mean Jacobian through layer 2)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=150)
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(float); ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
g0 = tools.greedy(w, X)
print(tools.trainability(w))
rows = []
for j in range(64):
    H = h1.copy(); H[:, j] = h1[:, j].mean()
    lg = np.tanh(H @ w["W2"].T + w["b2"]) @ w["W3"].T + w["b3"]
    flips = (lg.argmax(1)[ok] != g0[ok]).mean()
    J = (w["W3"] * (1 - h2 ** 2)[:, None, :]) @ w["W2"][:, j]      # [N,4] d logit / d h1_j
    Jc = J - J.mean(1, keepdims=True)
    rows.append((flips, j, h1[:, j].mean(), h1[:, j].std(), Jc.mean(0)))
rows.sort()
for f, j, m, s, d in rows[:16]:
    print(f"unit {j:2d} flips {f:.4f} mean {m:+.2f} std {s:.2f} push {d}")
