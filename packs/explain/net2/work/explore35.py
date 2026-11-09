from common import *
V = {n: v(X, n) for n in LANES}
A = 0.25*V['N0'] + V['N1'] + 0.75*V['S0'] + 0.75*V['S1']; B = V['N2'] + V['S2']
Cs = V['E1'] + 1.5*V['W1']; Cr = V['E0'] + V['W0']; D = V['E2'] + V['W2']
m3 = OK & (PH == 3); stay = A == 3
stay = (LG.argmax(1) == 3)
def tab(xv, yv, xb, yb, xl, yl, m):
    print(f"P(stay) rows {xl} bins {xb}, cols {yl} bins {yb}")
    for i in range(len(xb)-1):
        row = []
        for j in range(len(yb)-1):
            mm = m & (xv >= xb[i]) & (xv < xb[i+1]) & (yv >= yb[j]) & (yv < yb[j+1])
            row.append(f"{(stay & mm).sum():3d}/{mm.sum():<4d}" if mm.sum() else "   .    ")
        print(f"  {xb[i]:>4}-{xb[i+1]:<4}: " + " ".join(row))
tab(A, B, [0, 1, 2, 3, 4, 6, 9, 30], [0, 1, 2, 3, 4, 6, 13], "A", "B", m3)
tab(A, Cs, [0, 1, 2, 3, 4, 6, 9, 30], [0, 1, 2, 3, 4, 6, 20], "A", "Cs", m3)
tab(A + B, Cs + Cr, [0, 2, 4, 6, 8, 10, 14, 40], [0, 2, 4, 6, 8, 10, 14, 40], "A+B", "Cs+Cr", m3)
tab(A + B, D, [0, 2, 4, 6, 8, 10, 14, 40], [0, 1, 2, 3, 4, 13], "A+B", "D", m3)
