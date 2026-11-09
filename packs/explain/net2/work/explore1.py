import sys; sys.path.insert(0, '.')
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
print({k: (v.shape, v.dtype) for k, v in S.items()})
print(S['controller_names'] if 'controller_names' in S else None)
X = S['obs']; a = tools.greedy(w, X)
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
ctrl = S['controller']
print("n ok", ok.sum())
for c in np.unique(ctrl):
    m = (ctrl == c)
    print("controller", c, "n", m.sum(), "ok", (m & ok).sum())
# Transition matrix current phase -> net choice, ok states
for c in [None] + list(np.unique(ctrl)):
    m = ok if c is None else ok & (ctrl == c)
    M = np.zeros((4, 4), int)
    for p, q in zip(ph[m], a[m]): M[p, q] += 1
    print("ctrl", c); print(M)
# Not ok states
M = np.zeros((4, 4), int)
for p, q in zip(ph[~ok], a[~ok]): M[p, q] += 1
print("not ok"); print(M)
print(tools.trainability(w))
