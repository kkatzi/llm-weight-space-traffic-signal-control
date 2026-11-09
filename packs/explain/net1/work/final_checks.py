import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net1")
import numpy as np, tools
S = tools.states(); X = S["obs"]; w = tools.load("start_weights.npz"); L = tools.forward(w, X); net = L.argmax(1)
ok = X[:,44] > 0.5; srt = np.sort(L,1); gap = srt[:,-1]-srt[:,-2]
tot = X[:,12:24].sum(1)*3 + X[:,32:40].sum(1)*34.1
for f in ["work/rules_small.py", "work/rules_full.py"]:
    dec = tools._load_rules(f); r = np.array([dec(x) for x in X]); a = r==net
    print(f)
    for t in [1, 2]:
        print(f"   gap>={t}: share {(ok&(gap>=t)).sum()/ok.sum():.3f} agree {a[ok&(gap>=t)].mean():.3f};  gap<{t}: agree {a[ok&(gap<t)].mean():.3f}")
    print("   by load:", [(lo, round(a[ok&(tot>=lo)&(tot<hi)].mean(),3)) for lo,hi in [(0,10),(10,20),(20,30),(30,40),(40,200)]])
    # max scores sanity: saturated left vs through
