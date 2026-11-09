"""Hysteresis ('stay on current green') edit: add beta * s_p to row p of W3, where s_p is a sign-weighted
direction in layer-2 space built from correlations between each unit and the one-hot current-phase indicator.
Only correlations (statistics) are used; no fitting of weights."""
import sys; sys.path.insert(0, '/home/traffic-ntua/wsp_v3_agent_runs/patch_models/net1_haiku')
import numpy as np, tools
W = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"]
_, H2 = tools.hidden(W, X)
cur = X[:, 40:44].argmax(1)
ind = np.eye(4)[cur]
ok = X[:, 44] > 0.5
def corr_matrix(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    return (A.T @ B) / np.sqrt((A*A).sum(0)[:, None] * (B*B).sum(0)[None, :] + 1e-12)
C = corr_matrix(H2, ind)            # [64, 4]
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
S_dirs = np.where(np.abs(C) > THR, C, 0.0).T   # [4, 64]
S_dirs = S_dirs / np.linalg.norm(S_dirs, axis=1, keepdims=True)
def build(beta):
    w = {k: v.copy() for k, v in W.items()}
    w["W3"] = W["W3"] + beta * S_dirs
    return w
if __name__ == "__main__":
    for beta in [0, 1, 2, 3, 4, 6]:
        w = build(beta)
        pred = tools.greedy(w, X)
        sw = (pred[ok] != cur[ok]).mean()
        print(f"beta={beta}: switch share when ok {sw:.3f}  changed-vs-start {(pred!=tools.greedy(W,X)).mean():.3f}  "
              f"edit {tools.edit_size(w, W)['changed']}")
    print("nonzero per phase", (S_dirs != 0).sum(1))
