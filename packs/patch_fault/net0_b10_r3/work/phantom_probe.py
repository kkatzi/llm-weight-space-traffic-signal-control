import sys; sys.path.insert(0, '.')
import tools, numpy as np
S = tools.states(); X = S['obs'].astype(np.float64); C=S['controller']
w0 = tools.load("start_weights.npz")
def obs(phase, tg, extra={}):
    x = np.zeros(46); x[40+phase]=1; x[44]=1.0 if tg>=10/60 else 0.0; x[45]=tg
    for k,v in extra.items(): x[k]=v
    return x
# minimal W2 value to trigger phase 3 in the ORIGINAL net
for ph in [0,1,2]:
    for extra_name, extra in [("empty",{}),("W1u 6 halted",{31:0.18,39:0.18}),("W1u 12 halted",{31:0.35,39:0.35})]:
        out=[]
        for tg in [12/60, 30/60, 45/60, 1.0]:
            vmin=None
            for v in np.linspace(0,1,101):
                x=obs(ph,tg,extra); x[11]=v; x[23]=v
                if tools.forward(w0,x)[0].argmax()==3: vmin=round(v,2); break
            out.append(vmin)
        print(f"phase {ph} {extra_name:14s} min q_W2=d_W2 for phase3 at tg=12,30,45,60s: {out}")
# t_green distribution per phase under start_network & actuated
ph = X[:,40:44].argmax(1)
for c in range(3):
    m=C==c
    print(S['controller_names'][c], [ (p, np.round(np.quantile(X[m&(ph==p),45],[0.5,0.9,0.99]),2).tolist(), round(float((X[m&(ph==p),45]>=0.99).mean()),3)) for p in range(4)])
