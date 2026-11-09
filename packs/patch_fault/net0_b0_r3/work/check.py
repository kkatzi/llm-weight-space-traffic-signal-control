"""Checks of the repaired network (forward passes only)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np
import tools
from build import indicator, Q_E2, D_E2, Q_W2, D_W2, U

np.set_printoptions(precision=3, suppress=True, linewidth=160)
w0 = tools.load("start_weights.npz")
w = tools.load(os.path.join(HERE, "weights.npz"))           # as saved (float32 round trip)
wc = tools.load(os.path.join(HERE, "weights_copy_only.npz"))
S = tools.states()
X = S["obs"].astype(np.float64)
ctrl = S["controller"]
ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)
fire = indicator(X)

print("edit size:", tools.edit_size(w, w0))
print("indicator fires on", fire.mean().round(3), "of states; by source",
      [fire[ctrl == c].mean().round(3) for c in range(3)])

# implemented unit really behaves as the rule
h1, _ = tools.hidden(w, X)
print("unit U on where rule fires: min h", h1[fire, U].min().round(3), "; off where rule doesn't: max h",
      h1[~fire, U].max().round(3))

a_start = tools.greedy(w0, X)
a_copy = tools.greedy(wc, X)
a_new = tools.greedy(w, X)
Xfull = X.copy(); Xfull[:, Q_W2] = 1.0; Xfull[:, D_W2] = 1.0
a_tgt_fire = tools.greedy(w0, Xfull)       # full-info net, pocket full
Xcp = X.copy(); Xcp[:, Q_W2] = X[:, Q_E2]; Xcp[:, D_W2] = X[:, D_E2]
a_tgt_copy = tools.greedy(w0, Xcp)         # full-info net, pocket = E2 (identical to copy net)
target = np.where(fire, a_tgt_fire, a_tgt_copy)

for name, a in [("start(failed)", a_start), ("copy only", a_copy), ("copy+indicator", a_new)]:
    m1, m0 = ok & fire, ok & ~fire
    print(f"{name:15s} agree with 'ind|copy' target: indicator states {(a[m1] == target[m1]).mean():.3f}  "
          f"other states {(a[m0] == target[m0]).mean():.3f}  all {(a[ok] == target[ok]).mean():.3f}")
    print(f"{'':15s} honoured-decision shares by data source: " +
          "  ".join(f"src{c}:{np.bincount(a[ok & (ctrl == c)], minlength=4) / (ok & (ctrl == c)).sum()}"
                    for c in range(3)))

print("copy-net vs start-net on non-indicator honoured states: changed", (a_copy[ok & ~fire] != a_start[ok & ~fire]).mean().round(3))
print("final vs copy-net on non-indicator honoured states: changed", (a_new[ok & ~fire] != a_copy[ok & ~fire]).mean().round(3))

# per-phase look at indicator states: what does final net choose?
for p in range(3):
    m = ok & fire & (ph == p)
    if m.sum():
        print(f"indicator states in phase {p}: n={m.sum()} final choices {np.bincount(a_new[m], minlength=4)}"
              f" target {np.bincount(target[m], minlength=4)} start {np.bincount(a_start[m], minlength=4)}")

print("trainability start:", tools.trainability(w0))
print("trainability final:", tools.trainability(w))
