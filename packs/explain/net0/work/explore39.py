import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore36.py").read().split("variants = [")[0]
exec(src)
np.set_printoptions(precision=1, suppress=True, linewidth=250)
sf = mkS(); RS = np.array([sf(x) for x in X]); B = tie_biases(RS)
d = build(sf,B); r = np.array([d(x) for x in X])
SC = RS + np.array([B[c] for c in cur])
GF = np.array([[ (lambda l: [l["N0"]+l["N1"]+l["S0"]+l["S1"], l["N2"]+l["S2"], l["E0"]+l["E1"]+l["W0"]+l["W1"], l["E2"]+l["W2"], upEW(x)])(L(x)) ] for x in X])[:,0,:]
m = ok&~dead
for c in [0,3]:
    print("==== cur",c)
    C = np.zeros((4,4),int)
    for b_,c_ in zip(net[m&(cur==c)], r[m&(cur==c)]): C[b_,c_]+=1
    print(C)
    for (a,b) in [(1,2),(3,2),(2,1),(0,1),(1,0),(2,0)]:
        e = m&(cur==c)&(net==a)&(r==b); k = m&(cur==c)&(net==a)&(r==a); k2 = m&(cur==c)&(net==b)&(r==b)
        if e.sum()<5: continue
        print(f" net {a} rule {b}: n={e.sum()} mean loads(nsT,nsL,ewT,ewL,up) err {GF[e].mean(0)} | net=rule={a} {GF[k].mean(0)} | net=rule={b} {GF[k2].mean(0)}  ")
        print(f"     score gap S{a}-S{b} at errors: {np.round(np.percentile(SC[e,a]-SC[e,b],[10,50,90]),2)};  net logit gap {np.round(np.percentile(Lg[e,a]-Lg[e,b],[10,50,90]),2)}")
