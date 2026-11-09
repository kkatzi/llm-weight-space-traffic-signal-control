from derive_bonuses import *
V = lane_loads(X); U = upstream(X)
ns_thru = V['N1'] + 0.75*(V['S0'] + V['S1']) + 0.25*V['N0']; ns_left = V['N2'] + V['S2']
ew_str = V['E1'] + 1.5*V['W1']; ew_rgt = V['E0'] + V['W0']; ew_left = V['E2'] + V['W2']
up_ns = U['N1u'] + U['S1u']; up_ew = U['E0u'] + U['E1u'] + U['W0u'] + U['W1u']
S = np.stack([ns_thru - 0.45*ew_str + 0.2*ew_left + 0.1*up_ns,
              1.35*ns_left - 0.3*ew_rgt - 0.1*ew_left + 0.15*up_ns,
              1.1*ew_str + 0.75*ew_rgt + 0.5*ns_thru - 0.55*ns_left + 0.1*up_ew,
              1.55*ew_left + 0.5*ew_rgt - 0.2*ew_str - 0.25*ns_thru - 0.15*ns_left + 0.2*up_ew], 1)
# pair-specific offsets d[p][(i,j)] = b_i - b_j from tie medians
d = {}
for p in range(4):
    cands = [j for j in range(4) if j != p]
    for a_ in range(3):
        for b_ in range(a_+1, 3):
            i, j = cands[a_], cands[b_]
            m = OK & (PH == p) & (np.abs(LG[:, i] - LG[:, j]) < 0.75) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
            d[(p, i, j)] = -np.median(S[m, i] - S[m, j]); d[(p, j, i)] = -d[(p, i, j)]
def prefer(p, i, j, s):  # True if i preferred over j
    return s[i] - s[j] + d[(p, i, j)] > 0
for order_name, order in [("first pair = lowest two indices", None)]:
    r = np.zeros(len(X), int)
    for k in np.where(OK)[0]:
        p = PH[k]; cands = [j for j in range(4) if j != p]; s = S[k]
        a, b, c = cands
        w1 = a if prefer(p, a, b, s) else b
        r[k] = w1 if prefer(p, w1, c, s) else c
    report(r, label="pairwise tournament " + order_name)
    # all three pairwise: majority (Copeland), ties -> LSQ
    r2 = np.zeros(len(X), int)
    for k in np.where(OK)[0]:
        p = PH[k]; cands = [j for j in range(4) if j != p]; s = S[k]
        wins = {i: sum(prefer(p, i, j, s) for j in cands if j != i) for i in cands}
        top = [i for i in cands if wins[i] == 2]
        r2[k] = top[0] if top else r[k]
    report(r2, label="pairwise Copeland")
