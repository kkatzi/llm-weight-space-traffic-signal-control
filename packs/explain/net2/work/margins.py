"""How close to a tie is the network where the rules disagree with it?"""
import importlib.util
from common import *
def load(path):
    spec = importlib.util.spec_from_file_location("r", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.decide
for path in ["rules_small.py", "rules_full.py"]:
    dec = load(path); r = np.array([dec(x) for x in X])
    err = OK & (r != A); marg = LG[np.arange(len(X)), A] - LG[np.arange(len(X)), r]
    P = tools.probs(w, X)
    print(path, "disagreements", err.sum(), " network logit margin <1:", round((marg[err] < 1).mean(), 2), " <2:", round((marg[err] < 2).mean(), 2),
          " median margin", round(np.median(marg[err]), 2), " mean prob network gives the rule's pick:", round(P[np.arange(len(X)), r][err].mean(), 3))
