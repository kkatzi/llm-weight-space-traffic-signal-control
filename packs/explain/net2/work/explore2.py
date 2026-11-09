import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(linewidth=200, precision=2, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states(); X = S['obs'].astype(float); a = tools.greedy(w, X)
names = tools.names()
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
# mean features in ok states
print("feature means (ok states) / std")
for i, n in enumerate(names):
    print(f"{i:2d} {n:8s} mean {X[ok,i].mean():.3f} std {X[ok,i].std():.3f}  | all mean {X[:,i].mean():.3f}")
