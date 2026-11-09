"""Evaluate a rules file against the network on recorded states, broken down by controller and phase."""
import sys, importlib.util
from common import *
def load_rules(path):
    spec = importlib.util.spec_from_file_location("r", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
path = sys.argv[1]
mod = load_rules(path)
r = np.array([mod.decide(x) for x in X])
report(r, label=path)
for p in range(4):
    print("phase", p, "(network row, rule column), network's own runs only"); print(confusion(r, OK & (PH == p) & (CTRL == 0)))
print("stay decisions: network", (OK & (A == PH)).sum(), " rule", (OK & (r == PH)).sum(), " both", (OK & (r == PH) & (A == PH)).sum())
