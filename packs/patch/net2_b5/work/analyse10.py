import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
w = tools.load('start_weights.npz')
S = tools.states()
X = S['obs'].astype(float); E = S['episode']; T = S['time_s']; C = S['controller']
ph = X[:, 40:44].argmax(1); ok = X[:, 44] > 0.5
SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
g = tools.greedy(w, X)
q = X[:, :12] * 3
# for start net: track, per phase, waiting since last served; at each ok decision, list (wait, Q) for each phase and which was picked
m = C == 0
rows = []
for e in np.unique(E[m]):
    idx = np.where(m & (E == e))[0]
    last = {p: 0 for p in range(4)}
    for i in idx:
        p = ph[i]
        last[p] = T[i]
        if ok[i]:
            waits = [T[i] - last[r] for r in range(4)]
            rows.append((i, waits))
# how often is the longest-waiting phase (wait>=90) with queue skipped?
cnt = 0; tot = 0
for i, waits in rows:
    for r in range(4):
        Qr = q[i, SERVE[r]].sum()
        if waits[r] >= 90 and Qr > 0:
            tot += 1
            if g[i] != r:
                cnt += 1
                if cnt <= 15:
                    print(T[i], 'cur', ph[i], 'pick', g[i], 'waits', waits, 'Q', [round(q[i, SERVE[k]].sum()) for k in range(4)], 'q', q[i].round(0))
print('phase waited>=90s with queue, skipped', cnt, 'of', tot)
