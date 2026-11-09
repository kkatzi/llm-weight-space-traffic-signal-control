"""Worst-case bounds of the old pre-activations of the repurposed units over the whole input box
(traffic inputs in [0,1], one-hot phase, ok in {0,1}, t_green in [0,1]), so the K-trick margins can be checked."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
import build_full as B
w0 = tools.load("start_weights.npz")
wI = impute_w2(w0)   # rows of u_A/u_B as they are before the K terms are added
for u in [B.U_A, B.U_B]:
    r = wI["W1"][u]; b = wI["b1"][u]
    traffic = r[:40]
    hi = b + np.clip(traffic, 0, None).sum() + r[40:44].max() + max(r[44], 0) + max(r[45], 0)
    lo = b + np.clip(traffic, None, 0).sum() + r[40:44].min() + min(r[44], 0) + min(r[45], 0)
    print(f"layer-1 unit {u}: old pre-activation worst-case range [{lo:.2f}, {hi:.2f}]")
for j, u in [(B.J_A, B.U_A), (B.J_B, B.U_B)]:
    r = wI["W2"][j].copy(); r[u] = 0
    print(f"layer-2 unit {j}: old pre-activation worst-case range (all h1 in [-1,1]) [{wI['b2'][j]-np.abs(r).sum():.2f}, {wI['b2'][j]+np.abs(r).sum():.2f}]")
