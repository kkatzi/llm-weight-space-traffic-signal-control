import numpy as np, sys, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "work"))
import tools
from build import build
w0 = tools.load("start_weights.npz"); S = tools.states()
X = S["obs"].astype(np.float64); ctl = S["controller"]
ok = X[:, 44] > 0.5
ewE = (X[:, 17] == 0) & (X[:, 23] == 0); nsE = (X[:, 14] == 0) & (X[:, 20] == 0)
L0 = tools.forward(w0, X); a0 = L0.argmax(1)
Li = L0.copy(); Li[ok & ewE, 3] = -1e9; Li[ok & nsE, 1] = -1e9
a_int = Li.argmax(1)
print("intended changes (net episodes):", int(((a_int!=a0)&ok&(ctl==0)).sum()), " all:", int(((a_int!=a0)&ok).sum()))
for combo in [(50,1,38,56),(38,1,50,56),(50,56,38,1),(38,56,50,1)]:
    w = build(*combo, w0, S); a = tools.greedy(w, X)
    print(combo, "side effects net:", int(((a!=a_int)&ok&(ctl==0)).sum()), "all:", int(((a!=a_int)&ok).sum()),
          "| total changed honoured vs original net:", int(((a!=a0)&ok&(ctl==0)).sum()))
