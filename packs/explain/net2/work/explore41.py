from common import *
exec(open('explore38.py').read().split("S_small = ")[0])
S_full = np.load('S_tmp.npy')
srt = np.argsort(-LG, 1); top1, top2 = srt[:, 0], srt[:, 1]
def tie_offsets(S, tol=0.75, ctrl=None):
    out = {}
    for p in range(4):
        cands = [j for j in range(4) if j != p]
        eqs = []
        for a_ in range(3):
            for b_ in range(a_+1, 3):
                i, j = cands[a_], cands[b_]
                m = OK & (PH == p) & (np.abs(LG[:, i] - LG[:, j]) < tol) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
                if ctrl is not None: m &= CTRL == ctrl
                if m.sum() >= 5:
                    eqs.append((i, j, -np.median(S[m, i] - S[m, j]), m.sum()))   # b_i - b_j
        # solve small weighted least squares for b (cands), with sum zero
        Amat = []; y = []; wts = []
        for i, j, d, n in eqs:
            row = np.zeros(4); row[i] = 1; row[j] = -1; Amat.append(row); y.append(d); wts.append(np.sqrt(n))
        row = np.zeros(4); row[cands] = 1; Amat.append(row); y.append(0); wts.append(10)
        Amat = np.array(Amat) * np.array(wts)[:, None]; y = np.array(y) * np.array(wts)
        b = np.linalg.lstsq(Amat[:, cands], y, rcond=None)[0]
        bb = np.zeros(4); bb[cands] = b; bb[p] = np.nan
        out[p] = bb
        print(" phase", p, "pair offsets:", [(i, j, round(d, 2), n) for i, j, d, n in eqs], "->", bb.round(2))
    return out
def evaluate(S, Bn, label):
    r = np.zeros(len(X), int)
    for p in range(4):
        mm = OK & (PH == p)
        L = S[mm] + np.nan_to_num(Bn[p]); L[:, p] = -1e9; r[mm] = L.argmax(1)
    report(r, label=label); return r
Bt = tie_offsets(S_full)
evaluate(S_full, Bt, "full scores, tie-point offsets")
Bm = bonuses(S_full)
evaluate(S_full, Bm, "full scores, mean-matched offsets")
