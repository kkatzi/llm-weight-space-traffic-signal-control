from harness import *
OFF = {0: [0, 0.8, 0, 0], 1: [0, 0, 0.5, 0.5], 2: [0, -1.4, 0, 0], 3: [0, -1.7, -0.5, 0]}
def mk(kind, lw=1.8):
    def v(x, l):
        if kind=='d': return 3*x[ix['d_'+l]]
        if kind=='q': return 3*x[ix['q_'+l]]
        if kind=='qd': return 1.5*(x[ix['q_'+l]]+x[ix['d_'+l]])
    def decide(x):
        c = int(np.argmax(x[40:44]))
        s = np.array([v(x,'N0')+v(x,'N1')+v(x,'S0')+v(x,'S1'), lw*(v(x,'N2')+v(x,'S2')),
                      v(x,'E0')+v(x,'E1')+v(x,'W0')+v(x,'W1'), lw*(v(x,'E2')+v(x,'W2'))]) + np.array(OFF[c])
        s[c] = -1e9
        return int(np.argmax(s))
    return decide
for k in ['d','q','qd']: report(mk(k), k)
