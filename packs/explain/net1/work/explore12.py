from harness import *
from try3 import scores, OFF
import sys as _s
def pairdiag(p, a, b, band=2.0):
    m = np.where(ok & (cur==p) & np.isin(net,[a,b]))[0]
    sc = np.array([scores(X[i])+np.array(OFF[p]) for i in m])
    marg = sc[:,a]-sc[:,b]
    sel = np.abs(marg) <= band
    mm = m[sel]; ya = net[mm]==a
    print(f"cur {p}, {a} vs {b}: n near boundary {sel.sum()}, chose {a}: {ya.sum()}")
    # for each feature: mean when net chose a minus when chose b, controlling roughly by margin (subtract margin-matched)
    diffs = []
    for i in range(40):
        f = X[mm, i]
        diffs.append((f[ya].mean()-f[~ya].mean(), names[i]))
    print("   margin mean (chose a / chose b):", marg[sel][ya].mean().round(2), marg[sel][~ya].mean().round(2))
    print("   features with largest mean difference (a minus b):")
    print("   ", ", ".join(f"{nm}:{d:+.3f}" for d,nm in sorted(diffs, key=lambda t:-abs(t[0]))[:14]))
pairdiag(2,0,3); pairdiag(3,0,1); pairdiag(0,1,2); pairdiag(0,1,3); pairdiag(2,1,3); pairdiag(3,0,2)
