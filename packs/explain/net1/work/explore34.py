from trysmall import *
BA = np.array([[-99, 0.75, 0, 0.7],[0, 0.9, 0.26, 0.76],[0, -0.8, 2.0, 0.28],[0, -0.7, -0.5, 1.2]])
S_all = np.array([sc_small(x, up=0.1) for x in X]) + BA[cur]
pairs = {0:[(1,2),(1,3),(2,3)], 1:[(0,2),(0,3),(2,3)], 2:[(0,1),(0,3),(1,3),(2,0),(2,3)], 3:[(0,1),(0,2),(1,2),(3,0)]}
bins = np.arange(-2.0, 2.01, 0.5)
print("P(net=a | net in {a,b}) by rule margin; bins "+" ".join(f"{b:+.1f}" for b in bins[:-1]))
for c, pl in pairs.items():
    for a,b in pl:
        m = ok & (cur==c) & np.isin(net,[a,b])
        marg = S_all[:,a]-S_all[:,b]
        row = []
        for lo in bins[:-1]:
            mm = m & (marg>=lo) & (marg<lo+0.5)
            row.append(f"{(net[mm]==a).mean():.2f}/{mm.sum():<3d}" if mm.sum() else "   .    ")
        print(f"cur{c} {a}v{b}: "+" ".join(row))
