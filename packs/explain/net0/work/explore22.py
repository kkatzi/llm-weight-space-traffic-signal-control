import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore18.py").read()
src = "\n".join(l for l in src.splitlines() if not l.startswith("report("))
exec(src)
from sparsecheck import check
base = make(nsT1,ewT1,ewL0)
def with_keep(dec, lim=2.0):
    def d2(x):
        c = int(np.argmax(x[40:44]))
        if c==2:
            l = L(x)
            side = l["N0"]+l["N1"]+l["S0"]+l["S1"]+l["N2"]+l["S2"]+l["W2"]
            if l["E2"] < 0.5 and side <= lim and l["N2"]+l["S2"] <= 1.0:
                return 2
        return dec(x)
    return d2
report(base, "base(nsT1,ewT1)")
report(with_keep(base), "base + sparse keep rule")
for c in [2]:
    print("base", check(base,cur=c)); print("keep", check(with_keep(base),cur=c, verbose=True))
# which data states does keep rule fire on?
d2 = with_keep(base)
r1 = np.array([base(x) for x in X]); r2 = np.array([d2(x) for x in X])
ch = ok&(r1!=r2)
print("states changed:", ch.sum(), "net agrees with new:", (r2[ch]==net[ch]).sum(), "with old:", (r1[ch]==net[ch]).sum())
