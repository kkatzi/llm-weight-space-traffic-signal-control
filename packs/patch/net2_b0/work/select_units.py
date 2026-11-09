"""Pick which hidden units to repurpose: the combination that changes the fewest decisions
OTHER than the intended ones (statistics + forward passes only)."""
import numpy as np, sys, os, itertools
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build import build
w0 = tools.load("start_weights.npz"); S = tools.states()
X = S["obs"].astype(np.float64); ctl = S["controller"]
ok = X[:, 44] > 0.5
ewE = (X[:, 17] == 0) & (X[:, 23] == 0); nsE = (X[:, 14] == 0) & (X[:, 20] == 0)
L0 = tools.forward(w0, X)
# intended policy: original logits with 3 masked when EW pockets empty, 1 masked when NS pockets empty (only if ok)
Li = L0.copy(); Li[ok & ewE, 3] = -1e9; Li[ok & nsE, 1] = -1e9
a_int = Li.argmax(1)
ab = np.load(os.path.join(HERE, "work", "ablation.npz"))
c1 = list(np.argsort(ab["res1"])[:6]); c2 = list(np.argsort(ab["res2"])[:6])
best = []
for k3, k1 in itertools.permutations(c1, 2):
    for j3, j1 in itertools.permutations(c2, 2):
        w = build(k3, j3, k1, j1, w0, S)
        a = tools.greedy(w, X)
        bad_net = int(((a != a_int) & ok & (ctl == 0)).sum())
        bad_all = int(((a != a_int) & ok).sum())
        best.append((bad_net, bad_all, k3, j3, k1, j1))
best.sort()
for b in best[:10]: print(b)
