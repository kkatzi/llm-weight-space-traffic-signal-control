from harness import *
def mk(left_w=2.0, use='d', bias=None):
    def decide(x):
        g = lambda n: x[ix[use+'_'+n]]
        c = int(np.argmax(x[40:44]))
        s = np.array([g('N0')+g('N1')+g('S0')+g('S1'), left_w*(g('N2')+g('S2')),
                      g('E0')+g('E1')+g('W0')+g('W1'), left_w*(g('E2')+g('W2'))])
        if bias is not None: s = s + bias[c]
        s[c] = -1e9
        return int(np.argmax(s))
    return decide
for use in ['d','q']:
    for lw in [1.0, 1.5, 2.0]:
        report(mk(lw, use), f"use={use} left_w={lw}")
