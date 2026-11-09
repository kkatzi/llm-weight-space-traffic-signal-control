import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore36.py").read().split("variants = [")[0]
exec(src)
import tiebias
sf = mkS(); RS = np.array([sf(x) for x in X])
GF = np.array([[ (lambda l: [l["N0"]+l["N1"]+l["S0"]+l["S1"], l["N2"]+l["S2"], l["E0"]+l["E1"]+l["W0"]+l["W1"], l["E2"]+l["W2"]])(L(x)) ] for x in X])[:,0,:]
tot = GF.sum(1)
def tie_biases_mask(RS, mask):
    saved = tiebias.ok.copy()
    tiebias.ok[:] = saved & mask
    B = tiebias.tie_biases(RS, verbose=True)
    tiebias.ok[:] = saved
    return B
print("== all"); Ball = tie_biases_mask(RS, np.ones(len(X),bool))
for T in [10]:
    print("== light/medium tot <",T); Blo = tie_biases_mask(RS, tot<T)
    print("== heavy tot >=",T); Bhi = tie_biases_mask(RS, tot>=T)
    def dec(x):
        c = int(np.argmax(x[40:44]))
        if c==2 and hold(x): return 2
        l=L(x); t = l["N0"]+l["N1"]+l["S0"]+l["S1"]+l["N2"]+l["S2"]+l["E0"]+l["E1"]+l["W0"]+l["W1"]+l["E2"]+l["W2"]
        b = (Blo if t<T else Bhi)[c].copy()
        s = sf(x)+b
        if c!=2: s[c]=-99
        return int(np.argmax(s))
    r = report(dec, f"two-regime handicaps T={T}")
    m = ok&(ctrl==0)&~dead
    print("   per-cur own:", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
