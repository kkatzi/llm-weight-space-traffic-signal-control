"""Behaviour checks of the delivered network on recorded and synthetic states (forward passes only)."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools
from build import P, build, rule_scores

w = tools.load(os.path.join(HERE, "weights.npz"))
S = tools.states(); X = S["obs"].astype(np.float64)
A = S["act"]; C = S["controller"]
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
names = tools.names(); ix = {n: i for i, n in enumerate(names)}
np.set_printoptions(precision=3, suppress=True, linewidth=160)

a = tools.greedy(w, X); pr = tools.probs(w, X); sc = rule_scores(P, X)
print("trainability (all states):", tools.trainability(w))
print("trainability (ok states only):", tools.trainability(w, X[ok]))
print("trainability (ok=0 states only):", tools.trainability(w, X[~ok]))

mism = np.flatnonzero(a != sc.argmax(1))
srt = np.sort(sc, 1)
print("net/rule mismatches:", len(mism), " of which rule ties (gap<1e-9):", int((srt[mism, -1] - srt[mism, -2] < 1e-9).sum()),
      " mismatches at ok states:", int(ok[mism].sum()))

for c in range(2):
    m = ok & (C == c)
    keep = a[m] == ph[m]
    print(f"\n[{S['controller_names'][c]} states, ok=1, n={m.sum()}] net keep rate {keep.mean():.3f}; per current phase",
          [round(float(keep[ph[m] == p].mean()), 2) for p in range(4)])
    print("  net choice distribution:", np.bincount(a[m], minlength=4) / m.sum(),
          " recorded controller:", np.bincount(A[m], minlength=4) / m.sum())
    print("  agreement with recorded controller:", (a[m] == A[m]).mean())
    sw = (A[m] != ph[m]) & (a[m] != ph[m])
    print("  when both switch, same target:", (a[m][sw] == A[m][sw]).mean())
    # keep rate by t_green
    tg = X[m, 45] * 60
    for lo, hi in [(10, 13), (13, 18), (18, 23), (23, 40), (40, 61)]:
        mm = (tg >= lo) & (tg < hi)
        if mm.sum():
            print(f"  green {lo}-{hi}s: n {mm.sum():4d} net keep {np.mean(a[m][mm] == ph[m][mm]):.2f}")
    mp = pr[m].max(1)
    print("  max prob quantiles 10/50/90:", np.quantile(mp, [.1, .5, .9]))
    pk = pr[m][np.arange(m.sum()), ph[m]]
    print("  prob(keep) quantiles 10/50/90:", np.quantile(pk, [.1, .5, .9]))

# synthetic probes
def base(phase, tg=12):
    x = np.zeros(46); x[40 + phase] = 1; x[44] = 1.0 if tg >= 10 else 0.0; x[45] = min(tg / 60, 1); return x

def show(tag, x):
    lg = tools.forward(w, x)[0]; p = tools.probs(w, x)[0]
    print(f"  {tag:55s} logits {lg} probs {p} -> {lg.argmax()}  rule {rule_scores(P, x)[0]}")

print("\nsynthetic probes")
x = base(2); show("EW_through green, empty junction", x)
x = base(2); x[ix["q_N0"]] = x[ix["d_N0"]] = 1/3; show("EW_through green, 1 halting on N0", x)
x = base(2); x[[ix["q_N0"], ix["d_N0"], ix["q_S1"], ix["d_S1"]]] = 1/3; show("EW_through green, 2 halting NS through", x)
x = base(2); x[[ix["q_N0"], ix["d_N0"], ix["q_S1"], ix["d_S1"], ix["q_S0"], ix["d_S0"]]] = 1/3; show("EW_through green, 3 halting NS through", x)
x = base(2); x[[ix["q_E2"], ix["d_E2"], ix["q_W2"], ix["d_W2"]]] = 1.0; show("EW_through green, both EW pockets full", x)
x = base(2); x[[ix["q_E2"], ix["d_E2"], ix["q_W2"], ix["d_W2"]]] = 1.0; x[[ix["q_E0u"], ix["d_E0u"]]] = 6/34.13
show("  ... and 6 halting upstream E0u (EW through backlog)", x)
x = base(1); x[[ix["q_E0"], ix["d_E0"], ix["q_W0"], ix["d_W0"], ix["q_E1"], ix["d_E1"]]] = 1.0; show("NS_left green, 9 halting EW through", x)
x = base(0, 2); x[[ix["q_E0"], ix["d_E0"]]] = 1.0; show("NS_through green 2 s (ok=0), 3 halting E0", x)
