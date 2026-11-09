from common import *
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def scores(X):
    return np.stack([v(X,'N0')/3 + v(X,'N1') + v(X,'S0') + v(X,'S1'),
                     2*(v(X,'N2') + v(X,'S2')),
                     v(X,'E0') + v(X,'E1') + v(X,'W0') + 2*v(X,'W1'),
                     2*(v(X,'E2') + v(X,'W2'))], 1)
Sc = scores(X)
srt = np.argsort(-LG, 1); top1, top2 = srt[:, 0], srt[:, 1]
gap = LG[np.arange(len(X)), top1] - LG[np.arange(len(X)), top2]
# boundary offsets: mean of (s_i - s_j) when network nearly indifferent between i and j
print("current p, pair i-j: n, mean(s_i - s_j) at |L_i - L_j|<1 & top2 ; also median")
for p in range(4):
    for i in range(4):
        for j in range(i+1, 4):
            if p in (i, j): continue
            m = OK & (PH == p) & (np.abs(LG[:, i] - LG[:, j]) < 1.0) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
            if m.sum() < 5: print(p, i, j, m.sum()); continue
            d = Sc[m, i] - Sc[m, j]
            print(p, f"{i}-{j}", m.sum(), round(d.mean(), 2), np.median(d))
P = Sc.copy(); P[np.arange(len(P)), PH] = -99
r = P.argmax(1)
report(r, label="no offsets")
