"""Overdue events in detail: definition check, what the network wanted, ranks, queues."""
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

# definition check of overdue
pred = (ss >= 90) & (ph > 0)
print("overdue == (ss>=90 & halting>0):", (pred == over).mean(), " mismatches:", (pred != over).sum())
mm = np.where((pred != over).any(1))[0]
for i in mm[:10]:
    print("  mismatch", ep[i], t[i], ss[i], ph[i], over[i].astype(int), "green", gr[i], tg[i])
pred2 = (ss >= 90) & (ph > 0) & (np.arange(4)[None, :] != gr[:, None])
print("with p != green:", (pred2 == over).mean())
# ss>=90 but no halting
print("ss>=90 & halting==0 (not overdue):", ((ss >= 90) & (ph == 0)).sum())

# phase_halting vs queue obs: which lanes per phase
lanes = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
cap = np.array([22.4 / 7.5] * 12)
for p in range(4):
    approx = (X[:, lanes[p]] * cap[0]).sum(1)
    print(f"phase {p}: corr(ph, sum queue*cap) = {np.corrcoef(approx, ph[:, p])[0,1]:.3f}  mean ph {ph[:,p].mean():.2f} max {ph[:,p].max()}")

# For each forced overdue event (decision where only the overdue phase legal and ok and network wanted something else)
forced = np.where(over.any(1) & (legal.sum(1) == 1) & ~legal[idx, gr] & (act != np.argmax(legal, 1)))[0]
print("\nforced decisions:", len(forced))
rank_of_over = []
for i in forced:
    p = np.argmax(legal[i])
    order = np.argsort(-L[i])
    rank_of_over.append(int(np.where(order == p)[0][0]))
print("rank of overdue phase in network logits at forced decisions:", np.unique(rank_of_over, return_counts=True))
print("forced phase:", np.unique(np.argmax(legal[forced], 1), return_counts=True))
print("forced from green:", np.unique(gr[forced], return_counts=True))
print("halting of forced phase:", np.unique(ph[forced, np.argmax(legal[forced], 1)], return_counts=True))
print("network wanted:", np.unique(act[forced], return_counts=True))
# what network chose at the previous free decision (the one that started the current green)
print("\nprevious free decision before forced (start of current green): ss of the to-be-forced phase")
prev_ss = []
for i in forced:
    p = np.argmax(legal[i])
    j = i - 1
    while j >= 0 and ep[j] == ep[i] and not (legal[j].sum() == 4):
        j -= 1
    if j >= 0 and ep[j] == ep[i]:
        prev_ss.append((ss[j, p], ph[j, p], gr[j], act[j], tg[j], t[i] - t[j]))
prev_ss = np.array(prev_ss)
print("ss at prev free decision:", np.unique(prev_ss[:, 0], return_counts=True))
print("halting at prev free decision:", np.unique(prev_ss[:, 1], return_counts=True))
print("gap seconds:", np.unique(prev_ss[:, 5], return_counts=True))

# How many free decisions (4 legal) have some phase p != argmax with ss[p] in [70, 90) ?
free = legal.sum(1) == 4
print("\nfree decisions:", free.sum())
for lo in [60, 65, 70, 75, 80, 85]:
    m = free & ((ss >= lo) & (ss < 90) & (np.arange(4)[None, :] != act[:, None])).any(1)
    print(f" free with another phase ss in [{lo},90): {m.sum()}")
