import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore36.py").read().split("variants = [")[0]
exec(src)
from sparsecheck import sparse_states
sf = mkS(); RS = np.array([sf(x) for x in X]); B = tie_biases(RS)
d = build(sf,B)
for c in [0,1,3]:
    st = sparse_states(c, tgs=(0.2,), maxveh=2)
    Xs = np.array([s[2] for s in st]); nt = tools.greedy(w,Xs); rr = np.array([d(x) for x in Xs])
    print("cur",c,"agree",(nt==rr).mean().round(2))
    for (tg,combo,_),a,b in zip(st,nt,rr):
        if a!=b: print("    ",combo,"net",a,"rule",b, np.round(tools.forward(w,_)[0],1))
