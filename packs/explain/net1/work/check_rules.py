import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
S = tools.states(); X = S["obs"]; ctrl = S["controller"]; w = tools.load("start_weights.npz"); net = tools.greedy(w, X)
ok = X[:,44] > 0.5
for f in ["work/rules_small.py", "work/rules_full.py"]:
    fid = tools.fidelity(f); n = tools.count_comparisons(f)
    dec = tools._load_rules(f); r = np.array([dec(x) for x in X])
    per = {str(S["controller_names"][c]): round(float((r[ok&(ctrl==c)]==net[ok&(ctrl==c)]).mean()),4) for c in range(3)}
    cur = X[:,40:44].argmax(1)
    print(f, fid, "comparisons:", n, "per controller:", per,
          "stay rate rule/net:", round(float((r[ok]==cur[ok]).mean()),4), round(float((net[ok]==cur[ok]).mean()),4))
