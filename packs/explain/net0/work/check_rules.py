import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work"); sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import tools
from evalh import *
from sparsecheck import check
for f in sys.argv[1:]:
    print(f, "fidelity:", tools.fidelity(f), "comparisons:", tools.count_comparisons(f))
    dec = tools._load_rules(f)
    r = report(dec, "  breakdown")
    m = ok&(ctrl==0)&~dead
    print("   per-cur own (no stall):", [round(float((r[m&(cur==a)]==net[m&(cur==a)]).mean()),3) for a in range(4)])
    for c in range(4): print("   synthetic light traffic cur",c, check(dec,cur=c))
