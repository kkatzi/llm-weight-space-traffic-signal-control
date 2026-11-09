from try9 import *
import sys
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.65, 1.22, -0.12],[0.0, -0.5, -0.75, 0.75]])
S_all = np.array([scores9(x) for x in X]) + B[cur]
pairs = {0:[(1,2),(1,3),(2,3)], 1:[(0,2),(0,3),(2,3)], 2:[(0,1),(0,3),(1,3),(2,0),(2,3)], 3:[(0,1),(0,2),(1,2),(3,0)]}
bins = np.arange(-2.0, 2.01, 0.5)
print("P(net=a | net in {a,b}) by rule margin S_a - S_b (current offsets included); columns are bins "+" ".join(f"{b:+.1f}" for b in bins[:-1]))
for c, pl in pairs.items():
    for a,b in pl:
        m = ok & (cur==c) & np.isin(net,[a,b])
        marg = S_all[:,a]-S_all[:,b]
        row = []
        for lo in bins[:-1]:
            mm = m & (marg>=lo) & (marg<lo+0.5)
            row.append(f"{(net[mm]==a).mean():.2f}/{mm.sum():<3d}" if mm.sum() else "   .    ")
        print(f"cur{c} {a}v{b}: "+" ".join(row))
