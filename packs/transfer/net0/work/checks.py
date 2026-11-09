"""Checks on the delivered network work/weights.npz (forward passes only)."""
import numpy as np
from common import *
from proxy import score
from unstick_check import report
np.set_printoptions(precision=3, suppress=True, linewidth=200)
S = tools.states(); X = S['obs'].astype(float); ctrl = S['controller']
L2 = tools.states('states_L2.npz'); X2 = L2['obs'].astype(float)
src = tools.load('../source_weights_L2.npz')
w = tools.load(os.path.join(HERE, 'weights.npz'))
am = np.load(os.path.join(HERE, 'affine_map.npz')); M, c = am['M'], am['c']
ok = X[:, 52] > .5; tg = np.round(X[:, 53] * 60); cur = X[:, 48:52].argmax(1)

# 0. copied layers
s = np.load(os.path.join(os.path.dirname(HERE), 'source_weights_L2.npz')); z = np.load(os.path.join(HERE, 'weights.npz'))
print('0. W2 b2 W3 b3 identical to source:', all(np.array_equal(z[k], s[k]) for k in ['W2', 'b2', 'W3', 'b3']),
      ' shapes', {k: z[k].shape for k in z.files})

# 1. functional equivalence: net(x) == source(M x + c)
d = np.abs(tools.forward(w, X) - tools.forward(src, X @ M.T + c)).max()
print('1. max |logit(net, x) - logit(source, Mx + c)| =', f'{d:.1e}',
      ' decisions equal:', bool((tools.greedy(w, X) == tools.greedy(src, X @ M.T + c)).all()))
p0 = plain()
same = (tools.greedy(w, X) == tools.greedy(p0, X))
print('   unstick term inert at t_green = 12 s: decisions equal to plain mapping on',
      f'{same[ok & (tg == 12)].mean():.3f} of ok states with tg=12; on ok states with tg>12: {same[ok & (tg > 12)].mean():.3f}')

# 2. trainability
tr = tools.trainability(w)
print('2. trainability (delivered, L2W states):', {k: round(v, 4) for k, v in tr.items()})
print('   trainability (plain map, L2W states):', {k: round(v, 4) for k, v in tools.trainability(p0).items()})
print('   trainability (source, own L2 states):', {k: round(v, 4) for k, v in tools.trainability(src, X2).items()})
print('   |W1| max: delivered %.3f  source %.3f;  row-norm mean: delivered %.3f  source %.3f' %
      (np.abs(w['W1']).max(), np.abs(src['W1']).max(), np.linalg.norm(w['W1'], axis=1).mean(),
       np.linalg.norm(src['W1'], axis=1).mean()))

# 3. input regime
XM = X @ M.T + c
okm = ok & (tg >= 12)
print('3. mapped L2W inputs at honoured states: min %.3f max %.3f' % (XM[okm].min(), XM[okm].max()))
a1w = X @ w['W1'].T + w['b1']; a1s = X2 @ src['W1'].T + src['b1']
print('   layer-1 |pre-act| mean/q99: delivered-L2W %.3f/%.3f  source-L2 %.3f/%.3f' %
      (np.abs(a1w).mean(), np.quantile(np.abs(a1w), .99), np.abs(a1s).mean(), np.quantile(np.abs(a1s), .99)))

# 4. decisions on recorded states
a = tools.greedy(w, X)
print('4. pick distribution at honoured states', np.bincount(a[ok], minlength=4) / ok.sum(),
      ' switch share', round(float((a[ok] != cur[ok]).mean()), 3))
for k in [0, 1]:
    m = ctrl == k
    print(f'   queue-served proxy on {["fixed", "actuated"][k]:8s} states: net {score(X[m], a[m])}\n'
          f'                                      recorded controller {score(X[m], S["act"][m])}')
print('   source on its own L2 states', score(X2, tools.greedy(src, X2), l2=True))

# 5. synthetic single-movement probes
def probe(lanes, p, level, t=12):
    x = np.zeros(54)
    for l in lanes:
        x[IW['q_' + l]] = level; x[IW['d_' + l]] = level
    x[48 + p] = 1; x[52] = 1; x[53] = t / 60
    return x
cases = {'NS through': ['N0', 'N1', 'S0', 'S1'], 'NS left': ['N2', 'S2'],
         'EW through': ['E0', 'E1', 'E2', 'W0', 'W1', 'W2'], 'EW left': ['E3', 'W3'],
         'E left only': ['E3'], 'W through only': ['W0', 'W1', 'W2'], 'S left only': ['S2']}
print('5. single-movement probes: pick from current phase 0/1/2/3 (lane level 0.67 | 1.0)')
for name, lanes in cases.items():
    print(f'   {name:15s}', [int(tools.greedy(w, probe(lanes, p, .67))[0]) for p in range(4)],
          [int(tools.greedy(w, probe(lanes, p, 1.))[0]) for p in range(4)])

# 6. stuck-green check (frozen snapshots, see unstick_check.py)
print('6. frozen lone-vehicle snapshots')
report(p0, 'plain mapping (no unstick)')
report(w, 'delivered')
