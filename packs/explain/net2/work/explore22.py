from common import *
np.set_printoptions(linewidth=250, precision=2, suppress=True)
def loads(X):
    return np.stack([(v(X,'N0') + v(X,'N1') + v(X,'S0') + v(X,'S1'))/4,
                     (v(X,'N2') + v(X,'S2'))/2,
                     (v(X,'E0') + v(X,'E1') + v(X,'W0') + v(X,'W1'))/4,
                     (v(X,'E2') + v(X,'W2'))/2], 1)
Ld = loads(X)
B = np.array([[0, 0.7, 0.4, 0],
              [-1.5, 0, 1.0, 0],
              [1.0, 0, 0, 0],
              [1.2, 0, 2.0, 0]])
# tiny tie-break: prefer order from the grids
TB = np.array([[0, 0.02, 0.01, 0], [0.01, 0, 0.02, 0], [0.02, 0, 0, 0.01], [0.01, 0, 0.02, 0]])
Sc = Ld + B[PH] + TB[PH]
P = Sc.copy(); P[np.arange(len(P)), PH] = -99
r = P.argmax(1)
report(r, label="loads + offsets v2")
for p in range(4):
    print(p); print(confusion(r, OK & (PH == p)))
