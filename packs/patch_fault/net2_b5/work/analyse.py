"""Sensitivity of the logits to each input, on the recorded states (finite differences)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(float); c = S["controller"]
nm = tools.names()
ok = X[:, 44] > 0.5
base = tools.forward(w, X)
def sens(i, d=0.2):
    Xp = X.copy(); Xp[:, i] = np.clip(Xp[:, i] + d, 0, 1.2)
    return (tools.forward(w, Xp) - base) / d
print("mean d logit / d x (all states), columns = phases 0..3")
for i in [2, 5, 8, 11, 14, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39]:
    g = sens(i)
    gc = g - g.mean(axis=1, keepdims=True)
    print(f"{nm[i]:7s}", gc.mean(axis=0), " ok-states:", gc[ok].mean(axis=0))
