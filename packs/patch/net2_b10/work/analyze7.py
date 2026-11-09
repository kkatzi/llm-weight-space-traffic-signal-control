import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=200)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
h1, h2 = tools.hidden(w, X)
print("W3 norms per h2 unit (top):")
imp = np.abs(w["W3"]).sum(0) * h2.std(0)
order = np.argsort(imp)
print(" least important h2 units:", order[:10], imp[order[:10]].round(3))
print(" most important:", order[-10:], imp[order[-10:]].round(3))
# h2 units: mean per phase, std within phase
for p in range(4):
    sep = []
    for v in range(64):
        a = h2[ph == p, v]; b = h2[ph != p, v]
        sep.append((a.mean() - b.mean()) / (np.sqrt(0.5*(a.var() + b.var())) + 1e-6))
    sep = np.array(sep)
    o = np.argsort(-np.abs(sep))[:5]
    print(f"phase {p}: h2 units best separating:", o, sep[o].round(1), "means in/out", [(h2[ph==p, v].mean().round(2), h2[ph!=p, v].mean().round(2), h2[ph==p, v].std().round(2), h2[ph!=p, v].std().round(2)) for v in o[:3]])
for p in range(4):
    sep = []
    for u in range(64):
        a = h1[ph == p, u]; b = h1[ph != p, u]
        sep.append((a.mean() - b.mean()) / (np.sqrt(0.5*(a.var() + b.var())) + 1e-6))
    sep = np.array(sep)
    o = np.argsort(-np.abs(sep))[:5]
    print(f"phase {p}: h1 units best separating:", o, sep[o].round(1), [(h1[ph==p, u].mean().round(2), h1[ph!=p, u].mean().round(2), h1[ph==p, u].std().round(2), h1[ph!=p, u].std().round(2)) for u in o[:3]])
print("W1 phase columns magnitude: max abs per column", np.abs(w["W1"][:, 40:46]).max(0).round(2))
print("h1 std per unit:", h1.std(0).round(2))
