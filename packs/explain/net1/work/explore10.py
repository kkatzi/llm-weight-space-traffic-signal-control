from harness import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
n = lambda l: np.round(3*X[:,ix['d_'+l]]).astype(int)
nq = lambda l: np.round(3*X[:,ix['q_'+l]]).astype(int)
NT = n('N0')+n('N1')+n('S0')+n('S1'); NL = n('N2')+n('S2')
ET = n('E0')+n('E1')+n('W0')+n('W1'); EL = n('E2')+n('W2')
own = {0: NT, 1: NL, 2: ET, 3: EL}
U = lambda *ls: sum(X[:,ix['d_'+l]] for l in ls)
UNS = U('N0u','N1u','S0u','S1u'); UEW = U('E0u','E1u','W0u','W1u')
for p in [2,3,0,1]:
    m = ok & (cur==p)
    other = sorted(set(range(4))-{p})
    lo = L[m][:, other].max(1)
    stay = net[m]==p
    print(f"cur {p}: stays {stay.sum()} of {m.sum()}")
    print("  own count on stay:", np.bincount(own[p][m][stay], minlength=8), " on switch:", np.bincount(own[p][m][~stay], minlength=8))
    tot_other = sum(own[q][m] for q in other)
    print("  other-phases total count on stay: mean", tot_other[stay].mean() if stay.any() else None, "switch mean", tot_other[~stay].mean())
    print("  t_green on stay:", np.unique(np.round(X[m][stay,45]*60), return_counts=True))
    print("  upstream NS/EW on stay:", UNS[m][stay].mean() if stay.any() else None, UEW[m][stay].mean() if stay.any() else None, " switch:", UNS[m][~stay].mean(), UEW[m][~stay].mean())
