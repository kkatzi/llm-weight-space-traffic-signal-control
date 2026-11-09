from common import *
from collections import Counter
ep = S['episode']; t = S['time_s']
m0 = CTRL == 0
for e in np.unique(ep[m0])[:6]:
    m = m0 & (ep == e)
    idx = np.where(m)[0]; idx = idx[np.argsort(t[idx])]
    seq = []
    for k in idx:
        if OK[k]: seq.append(f"{PH[k]}>{A[k]}" if A[k] != PH[k] else f"{PH[k]}=")
    print("episode", e, "n", len(idx), "t range", t[idx].min(), t[idx].max())
    print(" ".join(seq[:120]))
# bigram of served phases in network's own runs
c = Counter()
for k in np.where(m0 & OK)[0]:
    c[(PH[k], A[k])] += 1
print(sorted(c.items()))
