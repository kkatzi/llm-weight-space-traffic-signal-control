import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=220)
exec(open('work/explore24.py').read().split("g = tools.greedy")[0])
ph = obs[:, 40:44].argmax(1); ok = obs[:, 44] > .5
I = np.arange(len(t))
Ntot = d.sum(1) + du.sum(1); Htot = q.sum(1) + qu.sum(1)
pos = {(ctl[i], ep[i], t[i]): i for i in range(len(t))}
def outcome(i, K, arr):
    s = 0.0
    for k in range(1, K+1):
        j = pos.get((ctl[i], ep[i], t[i] + 5*k), -1)
        if j < 0: return np.nan
        s += (arr[j] - arr[i]) * 5
    return s
sm = Wp.copy(); sm[I, ph] = -1e9
s = sm.argmax(1)
sw = ok & (act != ph) & (t < 3500)
# states where smith prefers a through phase and a left phase is the alternative taken or the through is taken
rows = []
for i in np.where(sw)[0]:
    if s[i] in (0, 2):
        rows.append((i, int(act[i] in (1, 3)), Wp[i, s[i]] - Wp[i, act[i]] if act[i] in (1,3) else np.nan))
rows = np.array(rows)
idx = rows[:, 0].astype(int); took_left = rows[:, 1] == 1
print("smith prefers through; controller took left:", took_left.sum(), " took smith-through:", ((act[idx] == s[idx])).sum())
for K in [6, 12, 24]:
    o = np.array([outcome(i, K, Ntot) for i in idx])
    for c in range(3):
        mc = ctl[idx] == c
        a = mc & took_left; b = mc & (act[idx] == s[idx])
        if a.sum() > 5 and b.sum() > 5:
            print(f"K={K} ctl{c}: left n={a.sum()} out={np.nanmean(o[a]):7.1f}±{np.nanstd(o[a])/np.sqrt(a.sum()):5.1f} startN={Ntot[idx[a]].mean():.1f} | through n={b.sum()} out={np.nanmean(o[b]):7.1f}±{np.nanstd(o[b])/np.sqrt(b.sum()):5.1f} startN={Ntot[idx[b]].mean():.1f}")
