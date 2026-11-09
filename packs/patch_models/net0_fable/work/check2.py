import numpy as np, tools, itertools
N = tools.names(); idx = {n:k for k,n in enumerate(N)}
w0 = tools.load("start_weights.npz"); w = tools.load("work/weights.npz")
def mk(phase, tg, lanes_q):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if tg>=10 else 0.0; x[45]=min(tg/60,1)
    for n,v in lanes_q.items(): x[idx["q_"+n]]=v; x[idx["d_"+n]]=v
    return x
serve = {"N0":0,"N1":0,"S0":0,"S1":0,"N2":1,"S2":1,"E2":3,"W2":3}
lanes = list(serve)
print("single car, from phase 2 (ok): lane -> new pick (logits 0,1,3)")
for ln in lanes:
    x = mk(2,12,{ln:1/3}); lg = tools.forward(w,x)[0]
    print(f"  {ln}: pick {lg.argmax()} want {serve[ln]}  logits {lg[[0,1,3]].round(2)}")
print("single car, from phases 0,1,3 (ok): where does the (old=new) net go?")
for p in [0,1,3]:
    row=[]
    for ln in lanes:
        if serve[ln]==p: continue
        x = mk(p,12,{ln:1/3}); row.append(f"{ln}->{tools.greedy(w,x)[0]}")
    print("  from", p, row)
print("two cars on same lane from phase 2:")
for ln in lanes:
    x = mk(2,12,{ln:2/3}); lg = tools.forward(w,x)[0]; print(f"  {ln}: pick {lg.argmax()} want {serve[ln]} logits {lg[[0,1,3]].round(2)}")
print("pairs of single cars from phase 2:")
for a,b in itertools.combinations(lanes,2):
    x = mk(2,12,{a:1/3,b:1/3}); lg = tools.forward(w,x)[0]; print(f"  {a}+{b}: pick {lg.argmax()} (serves {serve[a]},{serve[b]})")
