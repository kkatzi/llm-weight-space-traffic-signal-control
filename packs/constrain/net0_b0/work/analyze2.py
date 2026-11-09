"""Where do the rules bind in the recorded run? Counting statistics only (no fitting)."""
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
through = (gr % 2 == 0)

print("N decisions", N)
print("t_green values at decisions:", np.unique(tg, return_counts=True))
nleg = legal.sum(1)
print("number of legal phases:", np.unique(nleg, return_counts=True))
act_legal = legal[idx, act]
print("network request legal:", act_legal.mean(), " illegal count", (~act_legal).sum())

# classify illegal requests
cur_legal = legal[idx, gr]
mg = (~act_legal) & (nleg == 1) & cur_legal  # only current legal -> min green (or overdue==current?)
print("\nillegal, only current legal (min green):", mg.sum(),
      " of which ok=1 (would have switched w/o rules):", (mg & (X[:, 44] > 0.5)).sum())
print("   at t_green:", np.unique(tg[mg & (X[:, 44] > 0.5)], return_counts=True))
anyover = over.any(1)
print("decisions with some phase overdue:", anyover.sum())
ov_bind = (~act_legal) & anyover & (tg >= 10)
print("illegal & overdue exists & t_green>=10:", ov_bind.sum())
maxg = (~cur_legal)
print("current phase illegal (max green or overdue end):", maxg.sum(), " t_green there:", np.unique(tg[maxg], return_counts=True))
maxg_only = maxg & ~anyover
print("current illegal and no overdue (pure max green):", maxg_only.sum(), " network wanted current:", (maxg_only & (act == gr)).sum())

# illegal requests where several legal phases exist: shield's pick vs best-logit legal
multi = (~act_legal) & (nleg >= 2)
print("\nillegal with >=2 legal phases:", multi.sum())
for i in np.where(multi)[0][:40]:
    lg = np.where(legal[i], L[i], -np.inf)
    hal = np.where(legal[i], ph[i], -np.inf)
    print(f"ep{ep[i]} t={t[i]} gr={gr[i]} tg={tg[i]:.0f} act={act[i]} legal={legal[i].astype(int)} over={over[i].astype(int)} "
          f"halt={ph[i]} ss={ss[i]} best_logit_legal={lg.argmax()} most_halt={hal.argmax()} L={np.round(L[i],2)}")

# switching behaviour: chosen green durations (actual) per phase
print("\nactual green lengths (t_green at last decision before change), per phase")
for p in range(4):
    lens = []
    for e in np.unique(ep):
        ii = np.where(ep == e)[0]
        for a, b in zip(ii[:-1], ii[1:]):
            if gr[a] == p and gr[b] != p:
                lens.append(tg[a])
    lens = np.array(lens)
    print(p, len(lens), np.unique(lens, return_counts=True))

# overdue episodes: how often forced
print("\nfirst-overdue events per phase:")
for p in range(4):
    starts = over[1:, p] & ~over[:-1, p]
    print(p, starts.sum())
print("since_served max per phase:", ss.max(0))
