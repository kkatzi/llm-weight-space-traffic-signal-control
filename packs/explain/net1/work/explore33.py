from try9 import *
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.4, 1.1, -0.12],[0.0, -0.5, -0.75, 0.75]])
S_all = np.array([scores9(x) for x in X]) + B[cur]
r = S_all.argmax(1)
tot = X[:,12:24].sum(1)*3 + X[:,32:40].sum(1)*34.1
print("total vehicles (stop-line + upstream) quantiles at ok:", np.quantile(tot[ok],[.1,.25,.5,.75,.9,.99]).round(1))
for lo,hi in [(0,10),(10,20),(20,30),(30,40),(40,60),(60,200)]:
    m = ok & (tot>=lo)&(tot<hi)
    if m.sum(): print(f"  total in [{lo},{hi}): n={m.sum()} fidelity {(r[m]==net[m]).mean():.3f}  net stay rate {(net[m]==cur[m]).mean():.3f} rule stay {(r[m]==cur[m]).mean():.3f}")
# by time in episode
ts = S["time_s"]
for lo,hi in [(0,900),(900,1800),(1800,2700),(2700,3600),(3600,9000)]:
    m = ok & (ts>=lo)&(ts<hi)
    if m.sum(): print(f"  time [{lo},{hi}): n={m.sum()} fidelity {(r[m]==net[m]).mean():.3f}")
from collections import Counter
m = ok & (tot>=30)
c = Counter((int(cur[i]), int(net[i]), int(r[i])) for i in np.where(m & (r!=net))[0])
print("high-load errors (cur, net, rule):", c.most_common(12))
# for the top error type, compare upstream EW load
for (cc, nn, rr), k in c.most_common(4):
    e = m & (cur==cc) & (net==nn) & (r==rr)
    good = m & (cur==cc) & (net==rr) & (r==rr)
    ewUp = X[:,32:40][:, [2,3,6,7]].sum(1)*34.1; nsUp = X[:,32:40][:, [0,1,4,5]].sum(1)*34.1
    print(f"  ({cc},{nn},{rr}) n={k}: ewUp {ewUp[e].mean():.1f} vs agreeing-{rr} {ewUp[good].mean():.1f}; nsUp {nsUp[e].mean():.1f} vs {nsUp[good].mean():.1f}")
