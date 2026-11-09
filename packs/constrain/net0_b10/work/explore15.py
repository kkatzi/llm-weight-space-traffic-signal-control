import sys; sys.path.insert(0, ".")
import numpy as np, tools
S = tools.states()
X = S["obs"].astype(float); act = S["act"].astype(int); rs = S["rule_state"]; ep = S["episode"]; ts = S["time_s"]
legal = rs[:, 4:8] > 0.5; ph = rs[:, 12:16]; tg = rs[:, 16]; g = rs[:, 17].astype(int)
N = len(X); ok = X[:, 44] > .5
LANES = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
UPT = {0: [32, 33, 36, 37], 2: [34, 35, 38, 39]}
tot = ph.sum(1)
for lo, hi in [(0, 8), (8, 14), (14, 40)]:
    for T in (12, 17):
        m = np.where(ok & (tg == T) & np.isin(g, [0, 2]) & (tot >= lo) & (tot < hi))[0]
        if not len(m): continue
        sl = np.array([X[i, [12 + l for l in LANES[g[i]]]].sum() * 2.99 for i in m])
        sq = np.array([X[i, LANES[g[i]]].sum() * 2.99 for i in m])
        print(f"tot halting [{lo},{hi}) tg={T}: n={len(m)}  stop-line veh on green lanes mean {sl.mean():.2f}  halting {sq.mean():.2f}  share with >=2 veh {(sl >= 2).mean():.2f}")
# time profile: TTS-relevant halting by time-of-episode (15 min bins), averaged over episodes
for b in range(6):
    m = (ts >= b * 900) & (ts < (b + 1) * 900)
    print(f"t in [{b*15},{(b+1)*15}) min: n={m.sum()}, mean total halting {tot[m].mean():.1f}")
