from try9 import *
B = np.array([[-99, 0.4, 0.0, -0.3],[0.45, 0.5, 0.0, 0.2],[0.0, -0.4, 1.1, -0.12],[0.0, -0.5, -0.75, 0.75]])
S_all = np.array([scores9(x) for x in X]) + B[cur]
r = S_all.argmax(1)
for t in [0.5, 1, 2, 3, 5]:
    m = ok & (gap < t)
    print(f"gap<{t}: share of ok states {m.sum()/ok.sum():.3f}, rule agreement there {(r[m]==net[m]).mean():.3f}; agreement where gap>={t}: {(r[ok&~m]==net[ok&~m]).mean():.3f}")
