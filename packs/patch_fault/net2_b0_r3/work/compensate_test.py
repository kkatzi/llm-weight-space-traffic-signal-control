import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
from build import impute_w2
import build_full as B
w0 = tools.load("start_weights.npz")
S = tools.states(); X = S["obs"].astype(np.float64); t = S["time_s"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)
wI = impute_w2(w0)
aI = tools.greedy(wI, X)
condA = (ph == 2) & (X[:, 45] >= B.THETA_A)
condB = (ph == 3) & (X[:, 45] >= B.THETA_B) & (X[:, 5] == 0) & (X[:, 17] == 0)
other = ~(condA | condB) & ok
h1I, _ = tools.hidden(wI, X)

def comp(w, u, clamp, k):
    w = {kk: v.copy() for kk, v in w.items()}
    mean_u = h1I[:, u].mean()
    delta = w["W2"][:, u] * (mean_u - clamp)      # what layer 2 lost on average
    order = np.argsort(-np.abs(delta))[:k]
    w["b2"][order] += delta[order]
    return w

wF = B.build(w0, X)
print("no compensation: honoured changes outside conditions %.4f" % np.mean(tools.greedy(wF, X)[other] != aI[other]))
for k in [0, 4, 8, 16, 32, 64]:
    w = comp(wF, B.U_B, B.U_B_NORMAL, k)
    w = comp(w, B.U_A, 1.0, 0)
    print(f"compensate u_B top-{k}: {np.mean(tools.greedy(w, X)[other] != aI[other]):.4f}  changed {tools.edit_size(w, w0)['changed']}")
for k in [8, 16]:
    w = comp(comp(wF, B.U_B, B.U_B_NORMAL, k), B.U_A, 1.0, k)
    print(f"compensate both top-{k}: {np.mean(tools.greedy(w, X)[other] != aI[other]):.4f}  changed {tools.edit_size(w, w0)['changed']}")
