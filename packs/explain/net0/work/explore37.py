import sys; sys.path.insert(0, "/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work")
src = open("/home/traffic-ntua/wsp_v3_agent_runs/explain/net0/work/explore36.py").read().split("variants = [")[0]
exec(src)
np.set_printoptions(precision=2, suppress=True, linewidth=250)
names = tools.names()
sf = mkS(); RS = np.array([sf(x) for x in X]); B = tie_biases(RS, verbose=True)
SC = RS + np.array([B[c] for c in cur])
# for pairs (p,q) per cur: residual correlation of network logit diff with raw features, controlling for score diff
for c,p,q in [(0,1,2),(0,3,2),(1,0,2),(1,3,2),(2,0,3),(2,1,3),(3,0,1),(3,0,2),(3,1,2)]:
    m = ok&~dead&(cur==c)
    dl = Lg[m,p]-Lg[m,q]; ds = SC[m,p]-SC[m,q]
    k = np.cov(dl,ds)[0,1]/np.var(ds)
    res = dl - k*ds - (dl.mean()-k*ds.mean())
    cors = np.array([np.corrcoef(res, X[m,i])[0,1] if X[m,i].std()>1e-6 else 0 for i in range(46)])
    top = np.argsort(-np.abs(cors))[:6]
    print(f"cur {c} {p} vs {q} n={m.sum()} slope {k:.2f} resid sd {res.std():.2f}; top residual correlates:", [(names[i], round(float(cors[i]),2)) for i in top])
