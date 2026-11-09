from try7 import *
B = offsets(True, True)
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(scores(x) + B[c]))
r = report(dec, "model")
for p in range(4): confusion(r, p)
# stay analysis for cur 2: by best other score
m = np.where(ok & (cur==2))[0]
s = np.array([scores(X[i]) + B[2] for i in m])
best = np.max(s[:, [0,1,3]], 1)
stay = net[m]==2
print("cur2: stay rate by best other (score units) ")
for v in np.arange(0, 8, 0.5):
    mm = (best>=v)&(best<v+0.5)
    if mm.sum(): print(f"  {v:4.1f}: net stay {stay[mm].mean():.2f}  rule stay {(r[m][mm]==2).mean():.2f} n={mm.sum()}")
