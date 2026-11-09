from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
srt = np.argsort(-LG, 1); gap = LG[np.arange(len(X)), srt[:, 0]] - LG[np.arange(len(X)), srt[:, 1]]
V = {n: v(X, n) for n in LANES}
U = {n: X[:, 32+k]*34.13 for k, n in enumerate(['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u'])}
ns_thru = V['N1'] + 0.75*(V['S0'] + V['S1']) + 0.25*V['N0']
ns_left = V['N2'] + V['S2']
ew_thru = 0.75*(V['E0'] + V['W0']) + V['E1'] + 1.5*V['W1']
ew_str = V['E1'] + 1.5*V['W1']; ew_rgt = V['E0'] + V['W0']
ew_left = V['E2'] + V['W2']
up_ns = U['N1u'] + U['S1u']; up_ew = U['E0u'] + U['E1u'] + U['W0u'] + U['W1u']
SCALE = 1.13
def bonuses(S):
    out = {}
    for p in range(4):
        m_ref = OK & (PH == p) & (gap < 3)
        b = LG[m_ref].mean(0)/SCALE - S[m_ref].mean(0)
        b = b - b[[j for j in range(4) if j != p]].mean()
        out[p] = b
    return out
def evaluate(S, Bn, label):
    r = np.zeros(len(X), int)
    for p in range(4):
        mm = OK & (PH == p)
        L = S[mm] + Bn[p]; L[:, p] = -1e9; r[mm] = L.argmax(1)
    report(r, label=label); return r
S_small = np.stack([ns_thru + 0.2*ew_left,
                    1.35*ns_left,
                    1.25*ew_thru + 0.5*ns_thru - 0.55*ns_left,
                    1.55*ew_left + 0.45*ew_thru - 0.25*ns_thru - 0.15*ns_left], 1)
Bn = bonuses(S_small); evaluate(S_small, Bn, "4-group")
for p in range(4): print(p, Bn[p].round(2))
S_small2 = np.stack([ns_thru,
                    1.35*ns_left,
                    1.25*ew_thru + 0.5*ns_thru - 0.5*ns_left,
                    1.5*ew_left + 0.5*ew_thru], 1)
Bn2 = bonuses(S_small2); evaluate(S_small2, Bn2, "4-group simpler")
for p in range(4): print(p, Bn2[p].round(2))
S_full = np.stack([ns_thru - 0.45*ew_str + 0.2*ew_left + 0.1*up_ns,
                   1.35*ns_left - 0.3*ew_rgt - 0.1*ew_left + 0.15*up_ns,
                   1.1*ew_str + 0.75*ew_rgt + 0.5*ns_thru - 0.55*ns_left + 0.1*up_ew,
                   1.55*ew_left + 0.5*ew_rgt - 0.2*ew_str - 0.25*ns_thru - 0.15*ns_left + 0.2*up_ew], 1)
Bf = bonuses(S_full); evaluate(S_full, Bf, "5-group + upstream")
for p in range(4): print(p, Bf[p].round(2))
np.save('S_tmp.npy', S_full)
