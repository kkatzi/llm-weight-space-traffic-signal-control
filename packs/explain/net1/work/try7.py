from harness import *
srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
K = 2.4  # logits per halted NS-through vehicle
def groups(x):
    v = lambda l: 1.5*(x[ix['q_'+l]] + x[ix['d_'+l]])     # halted vehicle = 1, moving = 0.5
    u = lambda l: 34.1*x[ix['d_'+l]]                         # vehicles on an upstream lane
    return dict(nsT=v('N0')+v('S0')+v('S1'), n1w0=v('N1')+v('W0'), nsL=v('N2')+v('S2'),
                ewT=v('E0')+v('E1')+v('W1'), ewL=v('E2')+v('W2'),
                ewUp=u('E0u')+u('E1u')+u('W0u')+u('W1u'), nsUp=u('N0u')+u('N1u')+u('S0u')+u('S1u'))
def scores(x, cross=True, up=True):
    g = groups(x)
    s = np.array([g['nsT'] + 0.55*g['n1w0'], 1.65*g['nsL'], 0.875*g['ewT'] + 0.55*g['n1w0'], 1.4*g['ewL']])
    if cross:
        s += np.array([0.2*(g['nsL']+g['ewL']), 0, 0.3*(g['nsT']+g['ewL']), 0.2*(g['ewT']+g['nsT']+g['nsL'])])
    if up:
        s += np.array([0.03*g['nsUp'], 0.04*g['ewUp'], 0.09*g['ewUp'], 0.09*g['ewUp']])
    return s
def offsets(cross, up, band=2.0):
    B = np.zeros((4,4))
    for p in range(4):
        m = ok & (cur==p) & (gap<band)
        Sm = np.array([scores(x, cross, up) for x in X[m]])
        res = L[m]/K - Sm
        B[p] = res.mean(0) - res.mean(0)[[q for q in range(4) if q!=p]].mean()
    return B
if __name__ == "__main__":
    for cross in [False, True]:
        for up in [False, True]:
            B = offsets(cross, up)
            def dec(x, B=B, cross=cross, up=up):
                c = int(np.argmax(x[40:44])); return int(np.argmax(scores(x, cross, up) + B[c]))
            r = report(dec, f"cross={cross} up={up}")
            print(np.round(B,2))
