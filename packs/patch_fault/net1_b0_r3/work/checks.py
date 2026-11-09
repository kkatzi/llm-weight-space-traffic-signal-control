"""Checks of work/weights.npz on the faulty observations (x[11] = x[23] = 0, as deployed)."""
import os
from common import *
import stall, chain

w = tools.load(os.path.join(HERE, "weights.npz"))
print("edit size", tools.edit_size(w, W0))
ok = X[:, 44] > 0.5
ph = X[:, 40:44].argmax(1)
g0, g1 = tools.greedy(W0, X), tools.greedy(w, X)

print("\n1) honoured decisions (ok=1) per recorded controller: share changed, pick distribution old -> new")
for c, nm in enumerate(S["controller_names"]):
    m = (C == c) & ok
    print(f"   {nm:14s} changed {np.mean(g0[m] != g1[m]):.3f}  old {np.round(np.bincount(g0[m], minlength=4) / m.sum(), 3)}"
          f"  new {np.round(np.bincount(g1[m], minlength=4) / m.sum(), 3)}")

print("\n2) wasted holds (keep a phase whose stop-line lanes are empty while others queue)")
for nm, ww in [("start", W0), ("patched", w)]:
    m, wh = stall.wasted(ww, X)
    print(f"   {nm:8s} {wh.sum():5d} of {m.sum()}  (green >= 30 s: {(wh & (X[:, 45] >= 0.5)).sum()})")

print("\n3) gridlock-like states from the start net's failed episodes: phase 2 green >= 30 s, E/W through lanes empty")
gl = (ph == 2) & ok & (X[:, [15, 16, 21, 22]].sum(1) == 0) & (X[:, 45] > 0.5)
for nm, ww in [("start", W0), ("patched", w)]:
    L = tools.forward(ww, X[gl])
    print(f"   {nm:8s} picks {np.bincount(L.argmax(1), minlength=4)} of {gl.sum()}")

print("\n4) onset of the episode-3 gridlock (start-net run), logits start -> patched")
idx = np.where((C == 0) & (EP == 3) & (T >= 495) & (T <= 530))[0]
for i in idx:
    print(f"   t={T[i]}  {np.round(tools.forward(W0, X[i])[0], 2)} -> {np.round(tools.forward(w, X[i])[0], 2)}")

print("\n5) frozen-queue chains on sparse synthetic states (permanent holds)")
for nm, ww in [("start", W0), ("patched", w)]:
    print("  ", nm, end=": "); chain.run(ww, mirror=False)

print("\n6) trainability", tools.trainability(w))
