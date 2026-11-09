from try9 import *
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.65, 1.22, -0.12],[0.0, -0.5, -0.75, 0.75]])
S_all = np.array([scores9(x) for x in X]) + B[cur]
r = S_all.argmax(1)
m = ok & (cur==2) & np.isin(net,[1,3])
marg = S_all[:,1]-S_all[:,3]
# within cur2 / {1,3}: how does net's choice depend on rule margin, split by some candidate features
print("rule margin bins (S1-S3) -> P(net=1)")
for lo in np.arange(-3,3,0.5):
    mm = m & (marg>=lo) & (marg<lo+0.5)
    if mm.sum(): print(f" [{lo:4.1f},{lo+0.5:4.1f}): {(net[mm]==1).mean():.2f} n={mm.sum()}")
mm = m & (np.abs(marg)<1.0)
y = net[mm]==1
print("near boundary n=", mm.sum(), "P(net=1)=", y.mean())
for f in ['q_N2','q_S2','d_N2','d_S2','q_E2','q_W2','d_E2','d_W2','q_N0','q_S0','d_S1','q_N1','d_N1','q_W0','d_W0','d_E0','d_E1','d_W1','d_E1u','d_W1u','d_E0u','d_W0u','d_N1u','d_S1u','t_green']:
    a = X[mm, ix[f]]
    print(f"  {f:6s}: net=1 mean {a[y].mean():.3f}   net=3 mean {a[~y].mean():.3f}")
