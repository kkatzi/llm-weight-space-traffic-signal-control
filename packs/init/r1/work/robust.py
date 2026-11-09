"""Net vs rule agreement as a function of the rule's top-2 margin, on recorded and synthetic congested states."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, rule_scores
w = tools.load(os.path.join(HERE, "weights.npz"))
S = tools.states(); X = S["obs"].astype(np.float64)
rng = np.random.default_rng(0)

def report(tag, X):
    sc = rule_scores(P, X); r = sc.argmax(1); a = tools.greedy(w, X)
    s = np.sort(sc, 1); marg = s[:, -1] - s[:, -2]
    print(f"{tag}: n {len(X)} agree {np.mean(a == r):.4f}; max top score {s[:, -1].max():.1f}")
    for lo, hi in [(0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 1), (1, 3), (3, 100)]:
        m = (marg >= lo) & (marg < hi)
        if m.sum():
            print(f"   rule margin [{lo},{hi}) veh: n {m.sum():5d} agree {np.mean(a[m] == r[m]):.4f}")
    big = s[:, -1] > 20
    if big.sum():
        print(f"   states with top score > 20: n {big.sum()} agree {np.mean(a[big] == r[big]):.4f}; with margin >= 0.25: "
              f"{np.mean(a[big & (marg >= .25)] == r[big & (marg >= .25)]):.4f}")

report("recorded", X)
# synthetic congestion: scale upstream queue/density by 2-3x and fill stop-line lanes (all clipped to [0,1])
Xs = X.copy()
f = rng.uniform(1.5, 3.5, size=(len(X), 1))
Xs[:, 24:40] = np.clip(Xs[:, 24:40] * f + rng.uniform(0, 0.1, size=(len(X), 16)) * (rng.random((len(X), 16)) < .5), 0, 1)
Xs[:, 32:40] = np.maximum(Xs[:, 32:40], Xs[:, 24:32])   # density >= queue
st = rng.integers(0, 4, size=(len(X), 12)) / 3.0
Xs[:, 12:24] = np.maximum(Xs[:, 12:24], st); Xs[:, 0:12] = np.minimum(np.maximum(Xs[:, 0:12], st * (rng.random((len(X), 12)) < .7)), Xs[:, 12:24])
report("synthetic congested", Xs)
print("trainability on synthetic congested:", tools.trainability(w, Xs))
