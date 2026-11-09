"""Derive the phase-transition bonuses for a given score definition from the network's tie points:
for each current phase p and each pair of candidate phases (i, j), take the recorded states (min green elapsed)
where the network's logits for i and j are the top two and within 0.75 of each other, and use the median of
(score_i - score_j) there as the boundary offset b_j - b_i. The three pair offsets per phase are combined
(weighted by count) into one bonus per candidate, centred to mean zero. This is a statistic of the network's
switching decisions; nothing is optimised against fidelity."""
from common import *
srt = np.argsort(-LG, 1); top1, top2 = srt[:, 0], srt[:, 1]
def tie_offsets(S, tol=0.75, verbose=True):
    out = {}
    for p in range(4):
        cands = [j for j in range(4) if j != p]
        rows, y, wts = [], [], []
        for a_ in range(3):
            for b_ in range(a_+1, 3):
                i, j = cands[a_], cands[b_]
                m = OK & (PH == p) & (np.abs(LG[:, i] - LG[:, j]) < tol) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
                if m.sum() >= 5:
                    d = -np.median(S[m, i] - S[m, j])
                    row = np.zeros(4); row[i] = 1; row[j] = -1
                    rows.append(row); y.append(d); wts.append(np.sqrt(m.sum()))
                    if verbose: print(f"  phase {p}: b{i}-b{j} = {d:5.2f} (n={m.sum()})")
        row = np.zeros(4); row[cands] = 1; rows.append(row); y.append(0.0); wts.append(10.0)
        Am = np.array(rows) * np.array(wts)[:, None]; yy = np.array(y) * np.array(wts)
        b = np.linalg.lstsq(Am[:, cands], yy, rcond=None)[0]
        bb = np.zeros(4); bb[cands] = b
        out[p] = bb
        if verbose: print(f"  phase {p} bonuses: " + ", ".join(f"{j}: {bb[j]:+.2f}" for j in cands))
    return out
def lane_loads(X):
    return {n: v(X, n) for n in LANES}
def upstream(X):
    return {n: X[:, 32+k]*34.13 for k, n in enumerate(['N0u','N1u','E0u','E1u','S0u','S1u','W0u','W1u'])}
if __name__ == "__main__":
    V = lane_loads(X); U = upstream(X)
    ns_thru = V['N1'] + 0.75*(V['S0'] + V['S1']) + 0.25*V['N0']
    ns_left = V['N2'] + V['S2']
    ew_thru = 0.75*(V['E0'] + V['W0']) + V['E1'] + 1.5*V['W1']
    ew_str = V['E1'] + 1.5*V['W1']; ew_rgt = V['E0'] + V['W0']
    ew_left = V['E2'] + V['W2']
    up_ns = U['N1u'] + U['S1u']; up_ew = U['E0u'] + U['E1u'] + U['W0u'] + U['W1u']
    print("SMALL scores")
    S_small = np.stack([ns_thru + 0.2*ew_left,
                        1.35*ns_left,
                        1.25*ew_thru + 0.5*ns_thru - 0.55*ns_left,
                        1.55*ew_left + 0.45*ew_thru - 0.25*ns_thru - 0.15*ns_left], 1)
    tie_offsets(S_small)
    print("FULL scores")
    S_full = np.stack([ns_thru - 0.45*ew_str + 0.2*ew_left + 0.1*up_ns,
                       1.35*ns_left - 0.3*ew_rgt - 0.1*ew_left + 0.15*up_ns,
                       1.1*ew_str + 0.75*ew_rgt + 0.5*ns_thru - 0.55*ns_left + 0.1*up_ew,
                       1.55*ew_left + 0.5*ew_rgt - 0.2*ew_str - 0.25*ns_thru - 0.15*ns_left + 0.2*up_ew], 1)
    tie_offsets(S_full)
