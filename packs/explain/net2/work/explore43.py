from derive_bonuses import *
V = lane_loads(X); U = upstream(X)
ns_thru = V['N1'] + 0.75*(V['S0'] + V['S1']) + 0.25*V['N0']; ns_left = V['N2'] + V['S2']
ew_thru = 0.75*(V['E0'] + V['W0']) + V['E1'] + 1.5*V['W1']; ew_left = V['E2'] + V['W2']
up_ns = U['N1u'] + U['S1u']; up_ew = U['E0u'] + U['E1u'] + U['W0u'] + U['W1u']
# upstream coefficients for 4-group model from shared Taylor (explore37: UN col [.04,.11,-.07,-.08], UE col [-.07,-.09,.02,.14]) /1.13, shifted
S = np.stack([ns_thru + 0.2*ew_left + 0.1*up_ns,
              1.35*ns_left + 0.15*up_ns,
              1.25*ew_thru + 0.5*ns_thru - 0.55*ns_left + 0.1*up_ew,
              1.55*ew_left + 0.45*ew_thru - 0.25*ns_thru - 0.15*ns_left + 0.2*up_ew], 1)
B = tie_offsets(S)
r = np.zeros(len(X), int)
for p in range(4):
    mm = OK & (PH == p); L = S[mm] + B[p]; L[:, p] = -1e9; r[mm] = L.argmax(1)
report(r, label="4 groups + upstream (no stay rule)")
