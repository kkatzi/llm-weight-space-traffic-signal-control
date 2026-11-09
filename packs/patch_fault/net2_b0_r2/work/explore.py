"""Key statistics behind the design (recorded states + forward passes only)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

np.set_printoptions(precision=2, suppress=True, linewidth=200)
w0 = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64)
c = S["controller"]
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5
print("x11/x23 in recorded data: max", X[:, 11].max(), X[:, 23].max())

# 1. the network reacts to E2 and W2 roughly through their sum
lv = [0, 1 / 2.987, 2 / 2.987, 1.0]
m = (c == 0) & ok & (ph != 3)
T = np.zeros((4, 4))
for i, e in enumerate(lv):
    for j, v in enumerate(lv):
        Z = X[m].copy()
        Z[:, [5, 17]] = e
        Z[:, [11, 23]] = v
        T[i, j] = (tools.greedy(w0, Z) == 3).mean()
print("P(pick phase 3), start-net honoured non-phase-3 states; rows E2 level, cols W2 level\n", T)

# 2. mirror evidence: E1u halting vehicle outside phase 3 => E2 pocket full
for ci in range(3):
    mm = (c == ci) & (ph != 3) & (X[:, 27] > 0.01)
    print(f"ctrl {ci}: P(q_E2 = 1 | q_E1u > 0, phase != 3) = {(X[mm, 5] > 0.99).mean():.3f} (n={mm.sum()})")

# 3. W-side spillback while E2 not full: frequent under the failed start network, rare on the E side
for ci in range(3):
    mm = (c == ci) & (ph != 3)
    print(f"ctrl {ci}: P(q_W1u>0 & q_E2<1) = {((X[mm, 31] > 0) & (X[mm, 5] < 0.99)).mean():.3f}   "
          f"P(q_E1u>0 & q_E2<1) = {((X[mm, 27] > 0) & (X[mm, 5] < 0.99)).mean():.3f}")
