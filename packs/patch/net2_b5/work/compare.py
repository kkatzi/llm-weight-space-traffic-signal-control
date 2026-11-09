"""Compare decisions of an edited network with the original on the recorded states."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w0 = tools.load('start_weights.npz')
w = tools.load(sys.argv[1])
S = tools.states()
X = S['obs'].astype(float); C = S['controller']; T = S['time_s']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
g0 = tools.greedy(w0, X); g = tools.greedy(w, X)
q = X[:, :12]
others = q[:, [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]].sum(1)
trig = ok & (ph == 3) & (X[:, 17] + X[:, 23] == 0) & (others > 0)
print(tools.edit_size(w, w0))
print('ok states', ok.sum(), 'R1 condition true', trig.sum(), 'of which original holds 3:', (trig & (g0 == 3)).sum())
ch = ok & (g != g0)
print('changed decisions on ok states', ch.sum(), ' inside R1', (ch & trig).sum(), ' outside R1', (ch & ~trig).sum())
print('changed decisions on non-ok states (ignored by the signal)', (~ok & (g != g0)).sum())
print('inside R1, new picks', np.bincount(g[trig], minlength=4), ' original', np.bincount(g0[trig], minlength=4))
# in R1 states, do the new picks match original's best non-3 choice?
lg0 = tools.forward(w0, X)
alt = lg0[:, :3].argmax(1)
print('R1 states: new pick == original best among 0-2:', (g[trig] == alt[trig]).mean())
print('logit range new', np.abs(tools.forward(w, X)).max().round(2))
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
for i in np.where(ch & ~trig)[0][:10]:
    print('  outside change', C[i], T[i], 'cur', ph[i], 'old', g0[i], 'new', g[i], 'lg0', lg0[i].round(2), 'lg', tools.forward(w, X[i])[0].round(2))
