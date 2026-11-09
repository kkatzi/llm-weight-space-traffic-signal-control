from harness import *
# lane weights read off the per-vehicle pushes (explore17), in units of "one N0 vehicle"
W = {0: {'N0':1.0,'S0':1.0,'S1':0.7,'N1':0.3},
     1: {'N2':1.7,'S2':1.7},
     2: {'E0':0.7,'E1':1.1,'W1':0.85,'W0':0.35},
     3: {'E2':1.3,'W2':1.3}}
def raw_scores(x, kind='qd', up=True):
    s = np.zeros(4)
    for p, lw in W.items():
        for l, wt in lw.items():
            v = (x[ix['q_'+l]] + x[ix['d_'+l]]) * 1.5 if kind=='qd' else 3*x[ix['d_'+l]]
            s[p] += wt*v
    if up:
        ew = 34.1*sum(x[ix['d_'+l]] for l in ['E0u','E1u','W0u','W1u'])
        ns = 34.1*sum(x[ix['d_'+l]] for l in ['N1u','S0u','S1u'])
        s[2] += 0.12*ew; s[3] += 0.12*ew; s[1] += 0.06*ew
        s[0] += 0.05*ns
    return s
K = 2.5   # logits per unit (one halted N0 vehicle ~ 2.5 logits)
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
def build(kind, up):
    B = np.zeros((4,4))
    for p in range(4):
        m = ok & (cur==p) & (gap<2.0)
        Sm = np.array([raw_scores(x, kind, up) for x in X[m]])
        res = L[m] - K*Sm
        B[p] = (res - res.mean(1, keepdims=True)).mean(0) / K   # offsets in vehicle units
    return B
for kind in ['qd','d']:
    for up in [False, True]:
        B = build(kind, up)
        def dec(x, B=B, kind=kind, up=up, stay=True):
            c = int(np.argmax(x[40:44])); s = raw_scores(x, kind, up) + B[c]
            return int(np.argmax(s))
        report(dec, f"kind={kind} up={up}")
        if kind=='qd' and up: print(np.round(B,2))
