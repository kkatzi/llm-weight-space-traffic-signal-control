import numpy as np, tools
np.set_printoptions(precision=3, suppress=True, linewidth=160)
w = tools.load("start_weights.npz")
S = tools.states()
X, A, ep, t, c = S["obs"], S["act"], S["episode"], S["time_s"], S["controller"]
G = {0:[0,1,6,7], 1:[2,8], 2:[3,4,9,10], 3:[5,11]}
U = {0:[0,4], 1:[1,5], 2:[2,6], 3:[3,7]}  # upstream lanes feeding (index into 24..31)
ok = X[:,44]>0.5; ph = X[:,40:44].argmax(1)
g = tools.greedy(w,X)
m = (c==0)
dens = np.stack([X[:,12+np.array(G[p])].sum(1) for p in range(4)],1)   # served stop-line density per phase
que  = np.stack([X[:,np.array(G[p])].sum(1) for p in range(4)],1)
udens= np.stack([X[:,32+np.array(U[p])].sum(1) for p in range(4)],1)
sw = ok & (g!=ph) & m
print("switch decisions (net states):", sw.sum())
chosen_d = dens[np.arange(len(X)), g]
best_d = dens.max(1)
print("chosen phase served density = 0 while max other phase density>0:", int((sw & (chosen_d<0.01) & (best_d>0.01)).sum()))
print("chosen phase density < best by >=0.67 (2 cars):", int((sw & (best_d - chosen_d >= 0.66)).sum()))
print("chosen phase density < best by >=1.0 :", int((sw & (best_d - chosen_d >= 0.99)).sum()))
# per current phase, how does the net choose? relation to densities
for p in range(4):
    mm = sw & (ph==p)
    print(f"from phase {p}: n={mm.sum()}  chosen dist {np.bincount(g[mm],minlength=4)}  mean dens of phases at decision {dens[mm].mean(0)}  mean dens of chosen {chosen_d[mm].mean():.2f} best {best_d[mm].mean():.2f}")
# the general queue picture at stay decisions in phase 2
mm = ok&(ph==2)&(g==2)&m
print("stay in 2: n", mm.sum(), "mean dens per phase", dens[mm].mean(0), "udens", udens[mm].mean(0))
mm = ok&(ph==2)&(g!=2)&m
print("leave 2: n", mm.sum(), "mean dens per phase", dens[mm].mean(0), "udens", udens[mm].mean(0))
# when leaving phase 2, how many vehicles were on EW through lanes still (density)
print("dens served when leaving 2, pct:", np.percentile(dens[mm,2],[25,50,75,90]))
for p in [0,1,3]:
    mm = sw&(ph==p)
    print(f"dens served when leaving {p}, pct:", np.percentile(dens[mm,p],[25,50,75,90]), " and upstream dens pct:", np.percentile(udens[mm,p],[50,75,90]))
