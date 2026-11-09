"""Verify the hand-built network: detector/gate truth tables, margins, decision changes vs the original."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
from build2 import H1_UNITS, GATES, RULE_TARGET, SERVE
np.set_printoptions(linewidth=200, precision=2, suppress=True)
w0 = tools.load('start_weights.npz'); w = tools.load(sys.argv[1])
S = tools.states()
Xd = S['obs'].astype(float); C = S['controller']; T = S['time_s']

def synth(n=20000, seed=0):
    """random light/drain-like and random busy-like observations"""
    rng = np.random.default_rng(seed); X = np.zeros((n, 46))
    for i in range(n):
        kind = rng.integers(3)
        k = rng.integers(0, 4) if kind == 0 else rng.integers(0, 13)
        lanes = rng.choice(12, size=min(k, 12), replace=False)
        for l in lanes:
            v = rng.integers(1, 4) / 2.987
            X[i, 12 + l] = v; X[i, l] = v * rng.choice([0, 1, 1])
        if kind == 2:
            X[i, 24:40] = np.clip(rng.integers(0, 6, 16) / 34.13, 0, 1) * (rng.random(16) < 0.5)
            X[i, 24:32] = np.minimum(X[i, 24:32], X[i, 32:40])
        p = rng.integers(4); X[i, 40 + p] = 1
        tg = rng.choice([2, 7, 12, 17, 22, 27, 32, 47, 60]); X[i, 44] = float(tg >= 10); X[i, 45] = min(tg / 60, 1)
    return np.minimum(X, 1)

def truth(X):
    ok = X[:, 44] > 0.5; ph = X[:, 40:44].argmax(1); up0 = (X[:, 32:40] - X[:, 24:32]).sum(1) == 0
    D = np.stack([X[:, [12 + l for l in SERVE[p]]].sum(1) for p in range(4)], 1)
    oth3 = X[:, [l for l in range(12) if l not in SERVE[3]]].sum(1)
    t = {'R1': ok & (ph == 3) & (D[:, 3] == 0) & (oth3 > 0)}
    for p in range(4): t['E%d' % p] = ok & up0 & (D[:, p] == 0)
    return t

for label, X in [('recorded', Xd), ('synthetic', synth())]:
    pre1 = X @ w['W1'].T + w['b1']
    h1 = np.tanh(pre1); pre2 = h1 @ w['W2'].T + w['b2']
    tt = truth(X)
    print('==', label, len(X))
    for name, j in H1_UNITS.items():
        f = pre1[:, j] > 0
        err = (f != tt[name]).sum()
        marg = np.abs(pre1[:, j]).min()
        print('  detector %s (unit %d): fires %d, rule true %d, mismatches %d, min |pre| %.2f' % (name, j, f.sum(), tt[name].sum(), err, marg))
    for p, i in GATES.items():
        want = np.zeros(len(X), bool)
        for name, tp in RULE_TARGET.items():
            if tp == p: want |= tt[name]
        f = pre2[:, i] > 0
        print('  gate logit %d (unit %d): fires %d, want %d, mismatches %d, min |pre| %.2f' % (p, i, f.sum(), want.sum(), (f != want).sum(), np.abs(pre2[:, i]).min()))

# decisions vs original on recorded data
X = Xd; ok = X[:, 44] > 0.5
g0 = tools.greedy(w0, X); g = tools.greedy(w, X)
tt = truth(X); anyrule = np.zeros(len(X), bool)
for v in tt.values(): anyrule |= v
ch = ok & (g != g0)
print('ok decisions changed: %d of %d; where a rule fires: %d; elsewhere: %d' % (ch.sum(), ok.sum(), (ch & anyrule).sum(), (ch & ~anyrule).sum()))
# where a rule fires: is the new pick never a suppressed phase (unless all suppressed)?
supp = np.zeros((len(X), 4), bool)
for name, tp in RULE_TARGET.items(): supp[:, tp] |= tt[name]
bad = ok & anyrule & supp[np.arange(len(X)), g] & ~supp.all(1)
print('rule states where the new pick is a suppressed phase (not all suppressed):', bad.sum())
# where no rule fires: new pick equals the original?
print('non-rule ok states agreement with original: %.4f' % (g[ok & ~anyrule] == g0[ok & ~anyrule]).mean())
print(tools.edit_size(w, w0))
print('max |logit| recorded:', np.abs(tools.forward(w, X)).max().round(1))
