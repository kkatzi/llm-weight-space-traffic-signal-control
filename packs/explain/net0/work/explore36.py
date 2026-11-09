import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore35.py").read().split("for nm,sf in")[0]
exec(src)
def mkS(nsT_w=(1,1), nsL_w=(1,1), ewT_w=(1,1,1,1), ewL_w=(1,1), mv=0.5, up=0.1):
    def sf(x):
        q = x[0:12]*2.985; d = x[12:24]*2.985
        v = q + mv*(d-q)
        l = dict(zip("N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2".split(), v))
        nsT = nsT_w[0]*(l["N0"]+l["S0"]) + nsT_w[1]*(l["N1"]+l["S1"])
        nsL = nsL_w[0]*l["N2"]+nsL_w[1]*l["S2"]
        ewT = ewT_w[0]*l["E0"]+ewT_w[1]*l["E1"]+ewT_w[2]*l["W0"]+ewT_w[3]*l["W1"]
        ewL = ewL_w[0]*l["E2"]+ewL_w[1]*l["W2"]
        return np.array([nsT - 0.6*ewL, 1.1*nsL, 0.6*ewT - 0.5*nsL + up*upEW(x), ewL])
    return sf
variants = [("C plain", mkS()),
            ("C + nsL N2>S2", mkS(nsL_w=(1.2,0.8))),
            ("C + ewL E2>W2", mkS(ewL_w=(1.25,0.75))),
            ("C + nsT lane0>lane1", mkS(nsT_w=(1.2,0.8))),
            ("C + ewT no W0", mkS(ewT_w=(1.33,1.33,0,1.33))),
            ("C + no upstream", mkS(up=0.0)),
            ("C + moving 0.3", mkS(mv=0.3)),
            ("C + moving 0.7", mkS(mv=0.7)),
            ("C + nsL & ewL asym", mkS(nsL_w=(1.2,0.8), ewL_w=(1.25,0.75))),
           ]
m = ok&(ctrl==0)&~dead
for nm,sf in variants:
    RS = np.array([sf(x) for x in X]); B = tie_biases(RS)
    r = report(build(sf,B), nm)
    print("     per-cur own:", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
