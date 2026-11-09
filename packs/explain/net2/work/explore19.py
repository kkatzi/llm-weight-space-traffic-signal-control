from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
def scores(X):
    return np.stack([(v(X,'N0') + v(X,'N1') + v(X,'S0') + v(X,'S1'))/4,
                     (v(X,'N2') + v(X,'S2'))/2,
                     (v(X,'E0') + v(X,'E1') + v(X,'W0') + v(X,'W1'))/4,
                     (v(X,'E2') + v(X,'W2'))/2], 1)
B = np.array([[0, 0, 0, 0],[0, 0, 0.5, 0],[0.5, 0, 0, 0],[0, -0.75, 1.25, 0]])
Sc = scores(X) + B[PH]
P = Sc.copy(); P[np.arange(len(P)), PH] = -99
r = P.argmax(1)
names = tools.names()
cols = list(range(0, 12)) + list(range(12, 24)) + list(range(24, 40)) + [45]
def show(p, i, j):
    bad = OK & (PH == p) & (A == i) & (r == j)
    good = OK & (PH == p) & (A == j) & (r == j)
    good2 = OK & (PH == p) & (A == i) & (r == i)
    print(f"cur {p}: net {i} rule {j}: n={bad.sum()} ; rule-right-{j} n={good.sum()} ; both-{i} n={good2.sum()}")
    print("   feat   " + " ".join(f"{names[k][:6]:>6s}" for k in cols[:24]))
    for lab, m in [('bad', bad), (f'ok{j}', good), (f'ok{i}', good2)]:
        print(f"   {lab:6s} " + " ".join(f"{X[m, k].mean()*3:6.2f}" for k in cols[:24]))
    print("   feat   " + " ".join(f"{names[k][:6]:>6s}" for k in cols[24:]))
    for lab, m in [('bad', bad), (f'ok{j}', good), (f'ok{i}', good2)]:
        print(f"   {lab:6s} " + " ".join(f"{X[m, k].mean()*34:6.2f}" for k in cols[24:40]) + f" {X[m,45].mean()*60:6.1f}")
    print("   scores bad", Sc[bad].mean(0).round(2), " ok_j", Sc[good].mean(0).round(2), " ok_i", Sc[good2].mean(0).round(2))
show(3, 0, 2); show(3, 2, 0); show(2, 0, 3); show(2, 3, 0); show(0, 3, 1); show(1, 3, 2); show(1, 2, 3)
