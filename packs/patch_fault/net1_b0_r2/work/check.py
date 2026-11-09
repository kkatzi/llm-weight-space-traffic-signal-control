"""Checks on the delivered work/weights.npz (forward passes only, no simulation, no fitting)."""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import tools  # noqa: E402
from build import K, empty_states  # noqa: E402
from mirror_test import rot  # noqa: E402

np.set_printoptions(linewidth=200, precision=3, suppress=True)
w0 = tools.load("start_weights.npz")
w = tools.load(os.path.join(HERE, "weights.npz"))       # as delivered (float32 -> float64)
wi = {k: v.copy() for k, v in w0.items()}                # patch 1 only, for comparison
wi["W1"][:, 5] += w0["W1"][:, 11]; wi["W1"][:, 17] += w0["W1"][:, 23]

S = tools.states(); X = S["obs"].astype(float); c = S["controller"]; ep = S["episode"]
X[:, 11] = 0; X[:, 23] = 0                                # the fault (already 0 in the recordings)
ok = X[:, 44] > 0.5
vis = np.r_[0:11, 12:23, 24:40]

print("1. edit size:", tools.edit_size(w, w0))
assert tools.edit_size(w, w0)["changed"] <= 371

h1, _ = tools.hidden(w, X)
empty = X[:, vis].sum(1) == 0
print("2. watchdog unit %d on recorded states: non-empty max %.6f (n=%d), empty min %.6f (n=%d)"
      % (K, h1[~empty, K].max(), (~empty).sum(), h1[empty, K].min(), empty.sum()))

g0, gi, gf = tools.greedy(w0, X), tools.greedy(wi, X), tools.greedy(w, X)
print("3. honoured decisions (ok=1), by recording controller:")
for ci, nm in enumerate(S["controller_names"]):
    m = ok & (c == ci) & ~empty
    print("   %-13s n=%4d  final==patch1 %.4f   final==start %.4f   phase share start %s final %s"
          % (nm, m.sum(), (gf[m] == gi[m]).mean(), (gf[m] == g0[m]).mean(),
             np.bincount(g0[m], minlength=4) / m.sum(), np.bincount(gf[m], minlength=4) / m.sum()))

E = empty_states(); okE = E[:, 44] > 0.5
LE = tools.forward(w, E)
marg = LE[:, 3] - np.delete(LE, 3, axis=1).max(1)
print("4. all-visible-empty states (phase x t_green, ok=1): chosen", np.bincount(LE[okE].argmax(1), minlength=4),
      " min margin for phase 3: %.2f" % marg[okE].min())

# 5. one vehicle anywhere visible (and two-vehicle combos on upstream lanes), every phase / t_green:
#    watchdog must stay off and the decision must equal patch-1-only
rows = []
for p, kk in itertools.product(range(4), range(14)):
    tg = min(1.0, (2 + 5 * kk) / 60)
    base = np.zeros(46); base[40 + p] = 1; base[45] = tg; base[44] = 1.0 if tg >= 10 / 60 - 1e-9 else 0.0
    for li in list(range(11)):                          # stop-line lane li (not W2): one car, moving or halting
        for halting in (0, 1):
            x = base.copy(); x[12 + li] = 0.3348; x[li] = 0.3348 * halting; rows.append(x)
    for ui in range(8):                                  # upstream lane: one car
        for halting in (0, 1):
            x = base.copy(); x[32 + ui] = 0.0293; x[24 + ui] = 0.0293 * halting; rows.append(x)
R = np.array(rows)
hR, _ = tools.hidden(w, R)
print("5. single-vehicle synthetic states: n=%d, watchdog max %.6f, decisions equal to patch-1-only: %.4f"
      % (len(R), hR[:, K].max(), (tools.greedy(w, R) == tools.greedy(wi, R)).mean()))

# 6. gridlock-like recorded states: phase 2 green, E/W through stop-line lanes empty, E1u queued
ph = X[:, 40:44].argmax(1)
gl = (ph == 2) & ok & (X[:, [3, 4, 9, 10]].sum(1) == 0) & (X[:, 27] > 0.3)
print("6. gridlock-like states n=%d: start picks %s, final picks %s"
      % (gl.sum(), np.bincount(g0[gl], minlength=4), np.bincount(gf[gl], minlength=4)))

# 7. rotated W-spillback scenarios: agreement with full-detector reference (true W2 = 1)
m = (c > 0) & (X[:, 5] > 0.99) & (X[:, 27] >= 0.029) & ok
Rr = rot(S["obs"].astype(float)[m])
print("7. rotated W-spillback states (n=%d), agreement with full-detector decision:" % m.sum())
for v in (0.0, 0.3348, 0.6696, 1.0):
    T = Rr.copy(); T[:, 5] = v; T[:, 17] = v
    ref = tools.greedy(w0, T)
    D = T.copy(); D[:, 11] = 0; D[:, 23] = 0
    print("   E2=%.2f  start %.2f  final %.2f" % (v, (tools.greedy(w0, D) == ref).mean(), (tools.greedy(w, D) == ref).mean()))

# 8. precision: float64 build vs saved float32
from build import build  # noqa: E402
_, w64, _ = build()
print("8. float64 build vs saved float32: decision agreement on recorded states %.5f, max |logit diff| %.2e"
      % ((tools.greedy(w64, X) == gf).mean(), np.abs(tools.forward(w64, X) - tools.forward(w, X)).max()))
print("9. trainability (info only):", tools.trainability(w))

# 10. blockage unit G
from build import G  # noqa: E402
hG = h1[:, G]
fire = hG > 0
ph = X[:, 40:44].argmax(1)
sigW = (ph == 2) & ok & (X[:, 31] >= 0.029) & (X[:, 22] == 0)
print("10. blockage unit %d: |h| min %.6f; fires on %d recorded states; signature (computed directly) %d; mismatch %d"
      % (G, np.abs(hG).min(), fire.sum(), sigW.sum(), (fire != sigW).sum()))
for ci, nm in enumerate(S["controller_names"]):
    m = c == ci
    print("    %-13s fires %4d / %5d   decisions there: %s" % (nm, (fire & m).sum(), m.sum(),
                                                             np.bincount(gf[fire & m], minlength=4)))
hRG, _ = tools.hidden(w, R)
fR = hRG[:, G] > 0
print("    single-vehicle synthetic: G fires on %d (all = halting car on W1u under phase 2 & ok: %s); "
      "decisions equal to patch-1-only where G is off: %.4f"
      % (fR.sum(), bool(np.all((R[fR, 31] > 0) & (R[fR, 42] == 1) & (R[fR, 44] == 1))),
         (tools.greedy(w, R)[~fR] == tools.greedy(wi, R)[~fR]).mean()))

# 11. rotated W-blockage signature states: agreement with full-detector decision (W2 truly full)
m = (c > 0) & (ph == 2) & ok & (X[:, 27] >= 0.029) & (X[:, 16] == 0)
Rs = rot(S["obs"].astype(float)[m])
print("11. rotated W-blockage states (n=%d), agreement with full-detector decision:" % m.sum())
for v in (0.0, 0.3348, 0.6696, 1.0):
    T = Rs.copy(); T[:, 5] = v; T[:, 17] = v
    ref = tools.greedy(w0, T)
    D = T.copy(); D[:, 11] = 0; D[:, 23] = 0
    print("   E2=%.2f  start %.2f  patch1 %.2f  final %.2f   final picks %s"
          % (v, (tools.greedy(w0, D) == ref).mean(), (tools.greedy(wi, D) == ref).mean(),
             (tools.greedy(w, D) == ref).mean(), np.bincount(tools.greedy(w, D), minlength=4)))
