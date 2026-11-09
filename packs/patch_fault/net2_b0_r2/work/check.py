"""Checks on work/weights.npz (forward passes on recorded / synthetic states only)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tools  # noqa: E402

np.set_printoptions(precision=3, suppress=True, linewidth=200)
w0 = tools.load("start_weights.npz")
w = tools.load(os.path.join(HERE, "weights.npz"))
print("edit size:", tools.edit_size(w, w0))

S = tools.states()
X = S["obs"].astype(np.float64)
c = S["controller"]
ph = X[:, 40:44].argmax(1)
ok = X[:, 44] > 0.5


def with_w2(Xin, q, d):
    Z = Xin.copy()
    Z[:, 11] = q
    Z[:, 23] = d
    return Z


g_new = tools.greedy(w, X)
g_start = tools.greedy(w0, X)
g_copy = tools.greedy(w0, with_w2(X, X[:, 5], X[:, 17]))      # original net, W2 := E2
g_full = tools.greedy(w0, with_w2(X, 1.0, 1.0))               # original net, W2 full
h1, _ = tools.hidden(w, X)
s = h1[:, 50]
spill = (ph != 3) & (X[:, 31] > 0.01)

print("\nunit 50 activation: on (>0.5) share", (s > 0.5).mean(), " off (<-0.5) share", (s < -0.5).mean(),
      " mid", ((s > -0.5) & (s < 0.5)).mean())
print("detector on == (phase!=3 & q_W1u>0):", ((s > 0.5) == spill).mean())

for ci, nm in enumerate(S["controller_names"]):
    m = (c == ci) & ok
    print(f"\n[{nm}] honoured states {m.sum()}")
    for lab, g in [("start", g_start), ("start W2:=E2", g_copy), ("new", g_new)]:
        print(f"  {lab:14s} phase shares", np.bincount(g[m], minlength=4) / m.sum())
    ms = m & spill
    mn = m & ~spill
    print(f"  spill states {ms.sum()}: new==orig(W2 full) {np.mean(g_new[ms] == g_full[ms]):.3f}; "
          f"start==orig(W2 full) {np.mean(g_start[ms] == g_full[ms]):.3f}; copy==orig(W2 full) "
          f"{np.mean(g_copy[ms] == g_full[ms]):.3f}; new picks 3: {np.mean(g_new[ms] == 3):.3f}")
    print(f"  non-spill states {mn.sum()}: new==start W2:=E2 {np.mean(g_new[mn] == g_copy[mn]):.4f}")

print("\ntrainability start:", tools.trainability(w0))
print("trainability new:  ", tools.trainability(w))

# synthetic sweep: phase-3 pick probability vs E2 level and W1u queue, from recorded honoured non-phase-3 states
m = ok & (ph != 3)
print("\nP(pick 3) on honoured non-phase-3 states; rows E2 level (q=d), cols q_W1u = 0, 1, 3, 6 vehicles")
for e in [0, 1 / 2.987, 2 / 2.987, 1.0]:
    row = []
    for nv in [0, 1, 3, 6]:
        Z = X[m].copy()
        Z[:, 5] = e
        Z[:, 17] = e
        Z[:, 31] = nv / 34.13
        Z[:, 39] = max(Z[:, 39].max(), nv / 34.13) if False else np.maximum(Z[:, 39], nv / 34.13)
        row.append((tools.greedy(w, Z) == 3).mean())
    print(f"  E2={e:.2f}", np.round(row, 3))
m3 = ok & (ph == 3)
print("P(stay in 3) on honoured phase-3 states, q_W1u = 0, 1, 3, 6 vehicles:",
      [round(float((tools.greedy(w, np.c_[X[m3][:, :31], np.full(m3.sum(), nv / 34.13), X[m3][:, 32:]]) == 3).mean()), 3)
       for nv in [0, 1, 3, 6]],
      " start net:", [round(float((tools.greedy(w0, np.c_[X[m3][:, :31], np.full(m3.sum(), nv / 34.13), X[m3][:, 32:]]) == 3).mean()), 3)
                      for nv in [0, 1, 3, 6]])
