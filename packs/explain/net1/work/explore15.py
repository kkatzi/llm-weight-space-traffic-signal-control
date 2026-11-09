from harness import *
def grad(x):
    a1 = w["W1"]@x + w["b1"]; h1 = np.tanh(a1)
    a2 = w["W2"]@h1 + w["b2"]; h2 = np.tanh(a2)
    return w["W3"] @ ((1-h2**2)[:,None] * w["W2"]) @ ((1-h1**2)[:,None] * w["W1"])
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
for thr in [1e9, 2.0]:
    lin = {}
    for p in range(4):
        m = ok & (cur==p) & (gap < thr)
        G = np.mean([grad(x) for x in X[m]], axis=0)
        # intercept: mean over those states of (L - G x)
        c0 = (L[m] - X[m] @ G.T).mean(0)
        lin[p] = (c0, G)
    def dec(x):
        p = int(np.argmax(x[40:44])); c0, G = lin[p]
        return int(np.argmax(c0 + G @ x))
    report(dec, f"linear per cur, avg grad over states with gap<{thr}")
