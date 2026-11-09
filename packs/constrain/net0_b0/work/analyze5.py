"""Free decisions: what does the network pick, how decisive is it, how loaded is the picked phase.
Also per-episode demand indicators and forced counts."""
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

free = legal.sum(1) == 4
srt = np.sort(L, 1)
gap = srt[:, -1] - srt[:, -2]
print("free decisions", free.sum(), " stay picks:", (free & (act == gr)).sum())
sw = free & (act != gr)
print("logit gap top1-top2 at free switching decisions: quantiles",
      np.quantile(gap[sw], [0.05, 0.1, 0.25, 0.5, 0.75]))
thr = sw & (act % 2 == 0)
lft = sw & (act % 2 == 1)
print("free switch picks through:", thr.sum(), " left:", lft.sum())
print("halting on picked through phase:", np.unique(ph[thr, act[thr]], return_counts=True))
print("halting on picked left phase:", np.unique(ph[lft, act[lft]], return_counts=True))

# second choice when first is through, and gap
sec = np.argsort(-L, 1)[:, 1]
m = thr & (sec % 2 == 1)
print("through pick with left 2nd:", m.sum(), " gap quantiles", np.quantile(gap[m], [0.1, 0.25, 0.5]))
print("  of those gap<0.5:", (m & (gap < 0.5)).sum(), " gap<1:", (m & (gap < 1)).sum())

# per episode
for e in np.unique(ep):
    ii = ep == e
    forced = ii & over.any(1) & (legal.sum(1) == 1) & ~legal[idx, gr] & (act != np.argmax(legal, 1))
    tot_dens = X[ii][:, 12:24].sum(1).mean() + X[ii][:, 32:40].sum(1).mean()
    print(f"ep{e}: decisions {ii.sum()}, mean halting sum {ph[ii].sum(1).mean():.2f}, mean dens sum {tot_dens:.2f}, "
          f"forced {forced.sum()}, max tg {tg[ii].max():.0f}, max ss {ss[ii].max():.0f}")

# time profile of halting (by 15-min block) pooled
for b in range(5):
    ii = (t >= b * 900) & (t < (b + 1) * 900)
    print(f"block {b}: n {ii.sum()} halting sum {ph[ii].sum(1).mean():.2f}  overdue-any {over[ii].any(1).mean():.3f}")
