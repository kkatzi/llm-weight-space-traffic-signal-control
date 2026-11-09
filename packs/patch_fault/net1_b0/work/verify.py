"""Checks on work/weights.npz (forward passes only)."""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import tools, numpy as np
import build as B
np.set_printoptions(linewidth=200, precision=3, suppress=True)
w0 = tools.load('start_weights.npz')
w = tools.load(os.path.join(HERE, 'weights.npz'))           # float32 round trip
print('edit size', tools.edit_size(w, w0))
S = tools.states(); X = S['obs'].astype(float); c = S['controller']
X[:, 11] = 0; X[:, 23] = 0                                   # failed detector (already 0 in data)
ok = X[:, 44] > 0.5
Ym = X.copy(); Ym[:, 11] = Ym[:, 5]; Ym[:, 23] = Ym[:, 17]
ref = tools.greedy(w0, Ym)          # "ideal mirror": original weights fed with the proxy as input
new = tools.greedy(w, X)            # delivered network fed with the faulty input
fault = tools.greedy(w0, X)

def expected_triggers(X):
    ph = X[:, 40:44].argmax(1); tg = X[:, 45]; P = X[:, 24:32].sum(1)
    A = (tg + B.C_P * P >= 0.975)
    C = (P >= B.P0)
    return ph, A, C

ph, A, C = expected_triggers(X)
free = ~A & ~C
for ci in range(3):
    m = (c == ci) & ok
    print('%-13s honoured n=%d  agree(new, ideal mirror) on trigger-free states %.3f | agree(new, fault net) %.3f | share A %.3f C %.3f'
          % (S['controller_names'][ci], m.sum(), (new[m & free] == ref[m & free]).mean(), (new[m] == fault[m]).mean(), A[m].mean(), C[m].mean()))
    print('    action dist new', np.bincount(new[m], minlength=4) / m.sum(), ' fault', np.bincount(fault[m], minlength=4) / m.sum())
mA = A & ok
print('A-trigger states: %d, action == (p+1)%%4: %.3f' % (mA.sum(), (new[mA] == (ph[mA] + 1) % 4).mean()))
mC = C & ~A & ok
print('C-only states: %d, action in {p, p+1}: %.3f' % (mC.sum(), np.isin(new[mC] - ph[mC], [0, 1, -3]).mean()))

# gate saturation on recorded states
h1, h2 = tools.hidden(w, X)
for name, units in [('L1 A', B.A_UNITS), ('L1 C', B.C_UNITS), ('L2 A', B.GA), ('L2 C', B.GC)]:
    print(name, 'min |h| over states', np.abs(h1[:, units] if name.startswith('L1') else h2[:, units]).min(0).round(4))

# synthetic stress test: random backgrounds
rng = np.random.default_rng(0)
n = 200000
Z = rng.random((n, 46)) * (rng.random((n, 46)) < 0.5)
Z[:, 24:32] *= rng.random((n, 1)) ** 2            # varied upstream pressure
Z[:, 40:44] = 0; Z[np.arange(n), 40 + rng.integers(0, 4, n)] = 1
age = rng.choice(np.r_[np.arange(2, 70, 5), np.arange(0, 70, 5)], n)
Z[:, 45] = np.minimum(age / 60, 1); Z[:, 44] = (age >= 10).astype(float)
Z[:, 11] = 0; Z[:, 23] = 0
phz, Az, Cz = expected_triggers(Z)
P = Z[:, 24:32].sum(1); tg = Z[:, 45]
s = tg + B.C_P * P
clear = np.abs(s - 0.975) > 0.01                  # away from the soft edge
clearC = np.abs(P - B.P0) > 0.15
az = tools.greedy(w, Z)
mA = Az & clear & (Z[:, 44] > 0.5)
print('synthetic A states %d: action==(p+1)%%4 %.4f' % (mA.sum(), (az[mA] == (phz[mA] + 1) % 4).mean()))
mC = Cz & ~Az & clear & clearC
print('synthetic C-only states %d: action in {p,p+1} %.4f' % (mC.sum(), np.isin(az[mC] - phz[mC], [0, 1, -3]).mean()))
mF = ~Az & ~Cz & clear & clearC
Zm = Z.copy(); Zm[:, 11] = Zm[:, 5]; Zm[:, 23] = Zm[:, 17]
print('synthetic trigger-free states %d: agree with ideal mirror %.4f' % (mF.sum(), (az[mF] == tools.greedy(w0, Zm)[mF]).mean()))
print('max |logit| recorded', np.abs(tools.forward(w, X)).max().round(2))
print('trainability', tools.trainability(w))
