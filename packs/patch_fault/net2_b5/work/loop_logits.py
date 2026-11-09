import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tools
from impute import fold, W0
from drain_table import obs, AGES
np.set_printoptions(precision=2, suppress=True, linewidth=150)
a, f = 0.5, 0.25
aq = {5: a, 40: f, 41: f, 42: f}; ad = {17: a, 40: f, 41: f, 42: f}
w = fold(aq, ad)
for cfg in [{}, {"E0": 1}, {"E1": 1}, {"W0": 1}, {"W0": 2}, {"W1": 1}, {"E0": 1, "W0": 1}]:
    print(cfg)
    for p in [0, 1, 3]:
        L = tools.forward(w, np.array([obs(cfg, p, ag) for ag in AGES]))
        print("  p%d" % p, " ".join(f"[{' '.join(f'{v:5.1f}' for v in r)}]" for r in L[:4]))
