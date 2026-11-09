from harness import *
OFF = {0: [0, 0.8, 0, 0], 1: [0, 0, 0.5, 0.5], 2: [0, -1.4, 0, 0], 3: [0, -1.7, -0.5, 0]}
def scores(x, lw=1.8):
    v = lambda l: 3*x[ix['d_'+l]]
    return np.array([v('N0')+v('N1')+v('S0')+v('S1'), lw*(v('N2')+v('S2')),
                     v('E0')+v('E1')+v('W0')+v('W1'), lw*(v('E2')+v('W2'))])
def decide(x, off=True):
    c = int(np.argmax(x[40:44])); s = scores(x)
    if off: s = s + np.array(OFF[c])
    s[c] = -1e9
    return int(np.argmax(s))
r = report(lambda x: decide(x, False), "lw1.8 no offsets")
r = report(decide, "lw1.8 + offsets")
for p in range(4): confusion(r,p)
