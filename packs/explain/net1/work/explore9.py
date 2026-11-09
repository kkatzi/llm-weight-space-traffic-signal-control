from harness import *
from collections import Counter, defaultdict
n = lambda l: np.round(3*X[:,ix['d_'+l]]).astype(int)
NT = n('N0')+n('N1')+n('S0')+n('S1'); NL = n('N2')+n('S2')
ET = n('E0')+n('E1')+n('W0')+n('W1'); EL = n('E2')+n('W2')
def ceiling(keys, label):
    cells = defaultdict(Counter)
    for i in np.where(ok)[0]: cells[keys(i)][net[i]] += 1
    tot = sum(max(c.values()) for c in cells.values())
    print(f"{label}: best possible fidelity from these keys = {tot/ok.sum():.4f}, cells={len(cells)}")
ceiling(lambda i: (cur[i], NT[i], NL[i], ET[i], EL[i]), "cur + 4 group counts (density)")
nq = lambda l: np.round(3*X[:,ix['q_'+l]]).astype(int)
QT = nq('N0')+nq('N1')+nq('S0')+nq('S1'); QL = nq('N2')+nq('S2'); QET = nq('E0')+nq('E1')+nq('W0')+nq('W1'); QEL = nq('E2')+nq('W2')
ceiling(lambda i: (cur[i], NT[i], NL[i], ET[i], EL[i], QT[i], QL[i], QET[i], QEL[i]), "cur + group d and q counts")
ceiling(lambda i: (cur[i],)+tuple(np.round(3*X[i,:24]).astype(int)), "cur + all 24 stop-line values")
