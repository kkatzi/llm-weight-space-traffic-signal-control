import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0")
import numpy as np, tools
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(float); ctrl = S["controller"]; T=S["time_s"]; ep=S["episode"]
net = tools.greedy(w, X)
ok = X[:,44] > 0.5
cur = X[:,40:44].argmax(1)
dead = (ctrl==0)&(ep==2)&(T>=3625)
def report(dec, label):
    r = np.array([dec(x) for x in X])
    out = [label]
    for c,nm in enumerate(["own","fixed","act"]):
        m = ok&(ctrl==c)
        out.append(f"{nm}={(r[m]==net[m]).mean():.3f}")
    m = ok&(ctrl==0)&~dead
    out.append(f"own_nodead={(r[m]==net[m]).mean():.3f}")
    out.append(f"all_ok={(r[ok]==net[ok]).mean():.3f}")
    print("  ".join(out))
    return r
