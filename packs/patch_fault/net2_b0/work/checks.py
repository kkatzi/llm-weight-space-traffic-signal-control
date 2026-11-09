"""Checks of the delivered weights (forward passes only)."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=200)

w0 = tools.load(os.path.join(HERE, "start_weights.npz"))
w = tools.load(os.path.join(HERE, "work", "weights.npz"))
S = tools.states(); X = S["obs"].astype(float); c = S["controller"]
ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1)

print("1. edit size:", tools.edit_size(w, w0))
changed = {k: int((w[k] != w0[k]).sum()) for k in tools.KEYS}
print("   per tensor:", changed)
cols = np.where((w["W1"] != w0["W1"]).any(0))[0]
print("   changed W1 columns:", cols, [tools.names()[i] for i in cols])

# 2. exact equivalence: patched net on faulty inputs == original net on inputs with W2 := E2
Y = X.copy(); Y[:, 11] = X[:, 5]; Y[:, 23] = X[:, 17]
d = np.abs(tools.forward(w, X) - tools.forward(w0, Y)).max()
print("2. max |logit diff| patched(x) vs original(x with W2:=E2):", d,
      " decisions identical:", bool((tools.greedy(w, X) == tools.greedy(w0, Y)).all()))

# 3. decision shares on honoured (ok) states
a0 = tools.greedy(w0, X); a1 = tools.greedy(w, X)
print("3. honoured-decision shares   fault-net", np.bincount(a0[ok], minlength=4) / ok.sum(),
      " patched", np.bincount(a1[ok], minlength=4) / ok.sum(), " changed", (a0[ok] != a1[ok]).mean().round(3))
for p in range(4):
    m = ok & (ph == p)
    print(f"   current phase {p}: fault {np.bincount(a0[m], minlength=4)/m.sum()}  patched {np.bincount(a1[m], minlength=4)/m.sum()}")

# 4. no regression where the E/W left pockets are empty: patched == fault there (E2 = 0 => identical input)
m = ok & (X[:, 5] == 0) & (X[:, 17] == 0)
print("4. states with E2 empty:", m.sum(), " decisions identical:", bool((a0[m] == a1[m]).all()))

# 5. left phase never picked when both left pockets (E2 and the imputed W2) and the upstream lanes are empty?
m = ok & (X[:, 5] == 0) & (X[:, 17] == 0) & (X[:, 27] == 0) & (X[:, 31] == 0) & (ph != 3)
print("5. no EW-left demand visible, not in phase 3: n", m.sum(), " patched picks phase 3:", int((a1[m] == 3).sum()))

# 6. synthetic sweep: a typical phase-2 state with E2 at each level
base = np.median(X[ok & (ph == 2)], axis=0); base[40:44] = [0, 0, 1, 0]; base[44] = 1; base[45] = 0.3
base[11] = base[23] = 0
for v in [0, 1/3, 2/3, 1]:
    z = base.copy(); z[5] = z[17] = v
    print(f"6. phase-2 median state, E2={v:.2f}: patched logits {tools.forward(w, z)[0]}  pick {tools.greedy(w, z)[0]}"
          f"   (fault net pick {tools.greedy(w0, z)[0]})")

# 7. trainability numbers (information only)
print("7. trainability start  ", tools.trainability(w0))
print("   trainability patched", tools.trainability(w))
