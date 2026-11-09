import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
G = {0:[0,1,6,7], 1:[2,8], 2:[3,4,9,10], 3:[5,11]}
U = {0:[0,4], 1:[1,5], 2:[2,6], 3:[3,7]}
ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1); tg = np.round(X[:,45]*60).astype(int)
dens = np.stack([X[:,12+np.array(G[p])].sum(1) for p in range(4)],1)
que  = np.stack([X[:,np.array(G[p])].sum(1) for p in range(4)],1)
udens= np.stack([X[:,32+np.array(U[p])].sum(1) for p in range(4)],1)
for ci,cn in enumerate(S["controller_names"]):
    m = c==ci
    if ci==0: m = m & ~((ep==2)&(t>3640))   # drop stuck tail
    served_d = dens[np.arange(len(X)),ph]; served_u = udens[np.arange(len(X)),ph]
    other_q = que.sum(1) - que[np.arange(len(X)),ph]
    wasted = m & (served_d<0.01) & (served_u<0.01) & (other_q>0.2)
    print(f"== {cn}: decisions {m.sum()}, wasted-green decisions {wasted.sum()} ({wasted.mean()/m.mean()*100:.1f}%)  by tg:", {k:int((wasted&(tg==k)).sum()) for k in [2,7,12,17,22,27,32]}, " by phase:", [int((wasted&(ph==p)).sum()) for p in range(4)])
    # mean waiting (queue sum) per decision as proxy delay
    print("   mean total stop-line queue (cars):", (X[m,:12].sum(1)*3).mean().round(2), " mean upstream queue (cars):", (X[m,24:32].sum(1)*34.1).mean().round(2))
    # how many decisions at tg=2 (just switched) have served density 0 -> chose empty phase
    j = m&(tg==2)
    print("   fresh greens:", j.sum(), " with served stop-line dens==0:", int((j&(served_d<0.01)).sum()), " and upstream also 0:", int((j&(served_d<0.01)&(served_u<0.01)).sum()))
