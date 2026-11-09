from harness import *
from try3 import scores, OFF
for p in [2,3,1]:
    m = np.where(ok & (cur==p))[0]
    best = np.array([max((scores(X[i])+np.array(OFF[p]))[q] for q in range(4) if q!=p) for i in m])
    stay = net[m]==p
    print(f"cur {p}: stay fraction by best-other-score (vehicle units, rounded)")
    b = np.floor(best).astype(int)
    for v in range(0, 12):
        mm = b==v
        if mm.sum(): print(f"   {v:2d}: {stay[mm].mean():.2f} / {mm.sum()}")
