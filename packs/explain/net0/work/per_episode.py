import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
from evalh import *
for f in ["work/rules_small.py","work/rules_full.py"]:
    d = tools._load_rules(f); r = np.array([d(x) for x in X])
    out=[]
    for e in range(6):
        m = ok&(ctrl==0)&(ep==e)
        out.append(f"ep{e}: {(r[m]==net[m]).mean():.3f} (n={m.sum()})")
    print(f, " | ".join(out))
# summary stats used in the explanation
m = ok&(ctrl==0)
T = np.zeros((4,4),int)
for a,b in zip(cur[m&~dead],net[m&~dead]): T[a,b]+=1
print("own-run transitions excl. stall (rows current, cols chosen):\n", T)
print("share of own ok-states in the stall:", round(float((m&dead).sum()/m.sum()),3))
