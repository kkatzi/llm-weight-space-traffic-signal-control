from common import *
exec(open('explore38.py').read().split("S_small = ")[0])
S_full = np.load('S_tmp.npy'); Bf = bonuses(S_full)
r = np.zeros(len(X), int)
for p in range(4):
    mm = OK & (PH == p); L = S_full[mm] + Bf[p]; L[:, p] = -1e9; r[mm] = L.argmax(1)
err = OK & (r != A)
# network logit margin between its choice and rule's choice
marg = LG[np.arange(len(X)), A] - LG[np.arange(len(X)), r]
print("errors", err.sum(), "of", OK.sum())
print("network margin (chosen - rule's pick) quantiles on errors:", np.quantile(marg[err], [0.25, 0.5, 0.75, 0.9]).round(2))
print("share of errors with margin < 1:", (marg[err] < 1).mean().round(3), " <2:", (marg[err] < 2).mean().round(3))
print("errors that are stays (net picked current):", (err & (A == PH)).sum())
for p in range(4):
    print(p); print(confusion(r, OK & (PH == p) & (CTRL == 0)))
