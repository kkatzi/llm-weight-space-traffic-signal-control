from trysmall import *
STAY = {0: -99, 1: 0.9, 2: 2.0, 3: 1.2}
def offs(c):
    b = np.zeros(4)
    if c in (0, 1): b[1] += 0.7; b[3] += 0.7       # after an NS phase: left-turn phases +0.7
    else: b[1] -= 0.75                              # after an EW phase: NS-left -0.75
    b[c] = STAY[c]
    return b
def dec(x):
    c = int(np.argmax(x[40:44])); return int(np.argmax(sc_small(x, up=0.1) + offs(c)))
r = report(dec, "S-A + 3-part offsets")
for p in range(4): confusion(r,p)
