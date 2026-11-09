"""Checks on the delivered weights (forward passes only)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, tools
from collections import Counter
from build import ideal_logits, H_coef
np.set_printoptions(precision=2, suppress=True, linewidth=200)
CFG = dict(alpha=0.75, M=2.5, G=5.0)
w0 = tools.load("start_weights.npz"); w = tools.load(os.path.join(HERE, "weights.npz"))
print("1. edit size (reloaded float32 file):", tools.edit_size(w, w0))
print("   per tensor:", {k: int((w[k] != w0[k]).sum()) for k in tools.KEYS})
S = tools.states(); X = S["obs"].astype(np.float64); ok = X[:, 44] > .5; ctl = S["controller"]
lg0 = tools.forward(w0, X); lg1 = tools.forward(w, X); g0 = lg0.argmax(1); g1 = lg1.argmax(1)
gi = ideal_logits(w0, X, **CFG).argmax(1)
ph = X[:, 40:44].argmax(1); H = X @ H_coef().T; I = np.arange(len(X))
print("2. finite:", np.isfinite(lg1).all(), " max |logit| orig/new:", np.abs(lg0).max().round(2), np.abs(lg1).max().round(2))
for c, nm in enumerate(S["controller_names"]):
    m = ok & (ctl == c)
    print(f"3. {nm:13s} honoured decisions changed vs original: {(g1[m]!=g0[m]).mean():.3f}  (ideal edit {(gi[m]!=g0[m]).mean():.3f}; built vs ideal mismatch {(g1[m]!=gi[m]).mean():.3f})"
          f"  | all states changed {(g1[ctl==c]!=g0[ctl==c]).mean():.3f}")
m = ok & (ctl == 0)
ch = m & (g1 != g0)
print("4. on network's own honoured states: holds", (g0[m]==ph[m]).sum(), "->", (g1[m]==ph[m]).sum(),
      "; picks", np.bincount(g0[m], minlength=4), "->", np.bincount(g1[m], minlength=4))
dH = H[I, g1] - H[I, g0]
print("   changed decisions:", ch.sum(), " mean waiting-vehicle gain of new pick:", dH[ch].mean().round(2),
      " share with dH>0:", (dH[ch] > 0).mean().round(2), " orig logit gap median:", np.median((lg0[I, g0]-lg0[I, g1])[ch]).round(2))
# 5. rotation consistency
N = tools.names(); IX = {n: i for i, n in enumerate(N)}
R = np.array([IX[f"{n.split('_')[0]}_{ {'N':'S','S':'N','E':'W','W':'E'}[n.split('_')[1][0]] }{n.split('_')[1][1:]}"] if n.startswith(("q_", "d_")) else IX[n] for n in N])
for nm, ww in [("orig", w0), ("new", w)]:
    a = tools.greedy(ww, X); b = tools.greedy(ww, X[:, R])
    print(f"5. 180-degree rotation disagreement ({nm}) on honoured states: {(a[ok]!=b[ok]).mean():.3f}")
# 6. synthetic states
def state(phase, age, **kv):
    x = np.zeros(46); x[40 + phase] = 1; x[44] = float(age >= 10); x[45] = min(age/60, 1)
    for k, v in kv.items(): x[IX[k]] = v
    return x
tests = {
    "empty, ph2 age12": state(2, 12),
    "empty, ph0 age12": state(0, 12),
    "NS thr queue full, ph2": state(2, 12, q_N0=1, q_N1=1, q_S0=1, q_S1=1, d_N0=1, d_N1=1, d_S0=1, d_S1=1),
    "EW left pockets full, ph0": state(0, 12, q_E2=1, q_W2=1, d_E2=1, d_W2=1),
    "NS left pockets full, ph2": state(2, 12, q_N2=1, q_S2=1, d_N2=1, d_S2=1),
    "EW thr long upstream queue, ph0": state(0, 12, q_E0=1, q_E1=1, q_W0=1, q_W1=1, d_E0=1, d_E1=1, d_W0=1, d_W1=1, q_E0u=.3, q_W0u=.3, d_E0u=.35, d_W0u=.35),
    "everything full, ph3": state(3, 12, **{f"{p}_{a}{l}": 1 for p in "qd" for a in "NESW" for l in "012"}, **{f"{p}_{a}{l}u": .5 for p in "qd" for a in "NESW" for l in "01"}),
}
for k, x in tests.items():
    print(f"6. {k:32s} orig {tools.greedy(w0, x)[0]} P={tools.probs(w0, x)[0].round(2)}  new {tools.greedy(w, x)[0]} P={tools.probs(w, x)[0].round(2)}")
print("7. trainability orig:", {k: round(v, 3) for k, v in tools.trainability(w0).items()})
print("   trainability new: ", {k: round(v, 3) for k, v in tools.trainability(w).items()})
