import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from tiebias import *
from sparsecheck import check
def L(x):
    q = x[0:12]*2.985; d = x[12:24]*2.985
    return dict(zip("N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split(), (q+d)/2))
def upEW(x): return (x[[26,27,30,31]].sum()+x[[34,35,38,39]].sum())*34.13/2
def hold(x):
    l = L(x); nsT = l["N0"]+l["N1"]+l["S0"]+l["S1"]; nsL = l["N2"]+l["S2"]
    return (l["E2"] < 0.25) and (nsL < 0.25) and (l["W2"] <= 1) and (nsT <= 2 + l["W2"])
def scoresA(x):   # current grouped model
    l = L(x)
    nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); ewT = 1.33*(l["E0"]+l["E1"]+l["W1"])
    nsL = l["N2"]+l["S2"]; ewL = l["E2"]+l["W2"]
    return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL])
def scoresB(x):   # with N2>S2, E2>W2 asymmetries
    l = L(x)
    nsT = 1.2*(l["N0"]+l["S0"]) + 0.8*(l["N1"]+l["S1"]); ewT = 1.33*(l["E0"]+l["E1"]+l["W1"])
    nsL = 1.2*l["N2"]+0.8*l["S2"]; ewL = 1.25*l["E2"]+0.75*l["W2"]
    return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL])
def scoresC(x):   # plain group sums (simplest)
    l = L(x)
    nsT = l["N0"]+l["S0"]+l["N1"]+l["S1"]; ewT = l["E0"]+l["E1"]+l["W0"]+l["W1"]
    nsL = l["N2"]+l["S2"]; ewL = l["E2"]+l["W2"]
    return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + 0.1*upEW(x), ewL])
def build(sf, B, use_hold=True):
    def dec(x):
        c = int(np.argmax(x[40:44]))
        if use_hold and c==2 and hold(x): return 2
        s = sf(x) + B[c]
        if c!=2: s[c] = -99
        return int(np.argmax(s))
    return dec
for nm,sf in [("A grouped",scoresA),("B asym",scoresB),("C plain",scoresC)]:
    RS = np.array([sf(x) for x in X])
    B = tie_biases(RS, verbose=False)
    print(nm, {c: np.round(B[c],2).tolist() for c in B})
    r = report(build(sf,B), nm+" tie-biases")
    m = ok&(ctrl==0)&~dead
    print("   per-cur own:", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
