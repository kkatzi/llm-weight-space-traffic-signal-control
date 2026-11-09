"""Free decisions where a phase other than the network's pick will be forced right after the pick.
Compare queues of the pick c and the would-be-forced phase p. Also transition matrix."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools

S = tools.states()
w = tools.load("start_weights.npz")
X = S["obs"].astype(np.float64)
R = S["rule_state"]
ss, legal, over, ph, tg, gr = R[:, 0:4], R[:, 4:8] > 0.5, R[:, 8:12] > 0.5, R[:, 12:16], R[:, 16], R[:, 17].astype(int)
act = S["act"].astype(int)
ep, t = S["episode"], S["time_s"]
L = tools.forward(w, X)
N = len(act)
idx = np.arange(N)
np.set_printoptions(linewidth=200, precision=2, suppress=True)

# transitions of actual green
T = np.zeros((4, 4), int)
for a, b in zip(range(N - 1), range(1, N)):
    if ep[a] == ep[b] and gr[a] != gr[b]:
        T[gr[a], gr[b]] += 1
print("green transitions (row from, col to):\n", T)

print("multi-overdue decisions:", (over.sum(1) >= 2).sum(), " legal count there:",
      np.unique(legal[over.sum(1) >= 2].sum(1), return_counts=True))

free = legal.sum(1) == 4
cands = []
for i in np.where(free)[0]:
    c = act[i]
    if c == gr[i]:
        continue
    dur = 15 if c % 2 == 1 else 20
    for p in range(4):
        if p in (c, gr[i]):
            continue
        if ss[i, p] + dur >= 90:
            cands.append((i, p, c))
print("free decisions where p != pick will be (likely) forced after the pick:", len(cands))
for i, p, c in cands[:60]:
    print(f"ep{ep[i]} t={t[i]:4d} g={gr[i]} tg={tg[i]:.0f} pick={c} p={p} ss_p={ss[i,p]:.0f} halt={ph[i]} "
          f"L={L[i]} rank_p={int(np.where(np.argsort(-L[i])==p)[0][0])}")
arr = np.array([(ph[i, p], ph[i, c], L[i, c] - L[i, p]) for i, p, c in cands])
print("halting p vs c:  p>c:", (arr[:, 0] > arr[:, 1]).sum(), " p==c:", (arr[:, 0] == arr[:, 1]).sum(),
      " p<c:", (arr[:, 0] < arr[:, 1]).sum())
