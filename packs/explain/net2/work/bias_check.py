"""Do the rules choose each phase as often as the network (no systematic bias)? Also robustness checks."""
import importlib.util
from common import *
def load(path):
    spec = importlib.util.spec_from_file_location("r", path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.decide
for path in ["rules_small.py", "rules_full.py"]:
    dec = load(path)
    r = np.array([dec(x) for x in X])
    for c in [0, 1, 2]:
        m = OK & (CTRL == c)
        print(path, "ctrl", c, "network picks", np.bincount(A[m], minlength=4), "rule picks", np.bincount(r[m], minlength=4))
    # agreement per episode of the network's own runs
    ep = S['episode']
    accs = [(r[m] == A[m]).mean() for e in np.unique(ep[CTRL == 0]) for m in [OK & (CTRL == 0) & (ep == e)]]
    print("  per-episode agreement (network runs):", np.round(accs, 3))
    # agreement by time in episode (demand level proxy)
    t = S['time_s']
    for lo, hi in [(0, 900), (900, 1800), (1800, 2700), (2700, 3600), (3600, 6000)]:
        m = OK & (CTRL == 0) & (t >= lo) & (t < hi)
        print(f"  t {lo}-{hi}: n={m.sum()} agreement {(r[m] == A[m]).mean():.3f}")
    # robustness: random inputs, empty, full
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.random(46); x[40:44] = 0; x[40 + rng.integers(4)] = 1; x[44] = rng.integers(2)
        out = dec(x); assert out in (0, 1, 2, 3) and isinstance(out, int), out
    print("  random-input check ok")
