from common import *
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def scores(X):
    return np.stack([(v(X,'N0') + v(X,'N1') + v(X,'S0') + v(X,'S1'))/4,
                     (v(X,'N2') + v(X,'S2'))/2,
                     (v(X,'E0') + v(X,'E1') + v(X,'W0') + v(X,'W1'))/4,
                     (v(X,'E2') + v(X,'W2'))/2], 1)
B = np.array([[0, 0, 0, 0],
              [0, 0, 0.5, 0],
              [0.5, 0, 0, 0],
              [0, -0.75, 1.25, 0]])
Sc = scores(X) + B[PH]
P = Sc.copy(); P[np.arange(len(P)), PH] = -99
r = P.argmax(1)
report(r, label="per-lane avg + offsets")
for p in range(4):
    print(p); print(confusion(r, OK & (PH == p)))
# check boundary again
srt = np.argsort(-LG, 1); top1, top2 = srt[:, 0], srt[:, 1]
for p in range(4):
    for i in range(4):
        for j in range(i+1, 4):
            if p in (i, j): continue
            m = OK & (PH == p) & (np.abs(LG[:, i] - LG[:, j]) < 1.0) & (((top1 == i) & (top2 == j)) | ((top1 == j) & (top2 == i)))
            d = Sc[m, i] - Sc[m, j]
            print(" cur", p, f"{i}-{j}", m.sum(), "mean", round(d.mean(), 2))
