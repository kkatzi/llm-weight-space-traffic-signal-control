import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
def L(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985
    return dict(zip("N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split(), (q+d)/2))
def upEW(x): return (x[[26,27,30,31]].sum()+x[[34,35,38,39]].sum())*34.13/2
BIAS = {0:[-1.2,-2.3,0,-2.1], 1:[-2.4,-0.25,0,-1.5], 2:[-2.8,-1.9,0,-0.7], 3:[-1.9,-0.56,0,-0.05]}
def make(nsT_f, ewT_f, ewL_f, ns_in_2=0.0, bias=BIAS):
    def dec(x):
        c = int(np.argmax(x[40:44])); l = L(x)
        nsT = nsT_f(l); nsL = l["N2"]+l["S2"]; ewT = ewT_f(l); ewL = ewL_f(l)
        s = np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x) + ns_in_2*nsT, ewL]) + bias[c]
        if c!=2: s[c]=-99
        return int(np.argmax(s))
    return dec
nsT0 = lambda l: l["N0"]+l["N1"]+l["S0"]+l["S1"]
ewT0 = lambda l: l["E0"]+l["E1"]+l["W0"]+l["W1"]
ewL0 = lambda l: l["E2"]+l["W2"]
report(make(nsT0,ewT0,ewL0), "base")
report(make(nsT0,ewT0,ewL0,0.14), "base + 0.14 nsT in S2")
nsT1 = lambda l: 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"])
report(make(nsT1,ewT0,ewL0), "nsT lane0 heavier")
ewT1 = lambda l: 1.33*(l["E0"]+l["E1"]+l["W1"])
report(make(nsT0,ewT1,ewL0), "ewT ignore W0")
ewL1 = lambda l: 1.2*l["E2"]+0.8*l["W2"]
report(make(nsT0,ewT0,ewL1), "ewL E2 heavier")
report(make(nsT1,ewT1,ewL1,0.14), "all")
