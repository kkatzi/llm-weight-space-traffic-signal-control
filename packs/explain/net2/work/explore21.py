from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
def loads(X):
    return np.stack([(v(X,'N0') + v(X,'N1') + v(X,'S0') + v(X,'S1'))/4,
                     (v(X,'N2') + v(X,'S2'))/2,
                     (v(X,'E0') + v(X,'E1') + v(X,'W0') + v(X,'W1'))/4,
                     (v(X,'E2') + v(X,'W2'))/2], 1)
Ld = loads(X)
def table(p, i, j, other):
    # states with current p, network chose i or j; tabulate share choosing j by binned loads of i and j
    m = OK & (PH == p) & ((A == i) | (A == j))
    bi = np.clip(np.floor(Ld[:, i]), 0, 5).astype(int); bj = np.clip(np.floor(Ld[:, j]), 0, 5).astype(int)
    print(f"cur {p}: rows load {i} (floor), cols load {j}; cell = n_choose_{j}/n  [only states where net picked {i} or {j}]")
    for a_ in range(6):
        row = []
        for b_ in range(6):
            mm = m & (bi == a_) & (bj == b_)
            row.append(f"{(A[mm]==j).sum():3d}/{mm.sum():<3d}" if mm.sum() else "   .   ")
        print(f"  {a_}: " + " ".join(row))
table(1, 0, 2, 3); table(3, 0, 2, 1); table(2, 0, 3, 1); table(0, 1, 3, 2); table(0, 1, 2, 3); table(0, 2, 3, 1); table(2, 0, 1, 3); table(2, 1, 3, 0); table(3, 1, 2, 0); table(1, 2, 3, 0); table(3, 0, 1, 2)
print()
table(1, 0, 3, 2)
# stays
for p in range(4):
    m = OK & (PH == p) & (A == p)
    if m.sum(): print("stay", p, m.sum(), "loads mean", Ld[m].mean(0).round(2), "max other", np.sort(Ld[m], 1)[:, -1].round(2)[:30])
