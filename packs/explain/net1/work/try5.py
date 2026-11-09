from harness import *
OFF = {0: [0, 0.8, 0, 0], 1: [0, 0, 0.5, 0.5], 2: [0, -1.4, 0, 0], 3: [0, -1.7, -0.5, 0]}
def mk(kind='d', lw=1.8, wN1=0.0, wW0=0.0, tw=1.0, off=OFF):
    def v(x, l):
        if kind=='d': return 3*x[ix['d_'+l]]
        return 1.5*(x[ix['q_'+l]]+x[ix['d_'+l]])
    def decide(x):
        c = int(np.argmax(x[40:44]))
        s = np.array([tw*(v(x,'N0')+wN1*v(x,'N1')+v(x,'S0')+v(x,'S1')), lw*(v(x,'N2')+v(x,'S2')),
                      tw*(v(x,'E0')+v(x,'E1')+wW0*v(x,'W0')+v(x,'W1')), lw*(v(x,'E2')+v(x,'W2'))]) + np.array(off[c])
        s[c] = -1e9
        return int(np.argmax(s))
    return decide
if __name__ == "__main__":
    report(mk('d', 1.8, 1, 1), "d all lanes")
    report(mk('d', 1.8, 0, 0), "d drop N1,W0")
    report(mk('d', 1.8, 0.5, 0.5), "d half N1,W0")
    report(mk('qd', 1.8, 0, 0), "qd drop N1,W0")
