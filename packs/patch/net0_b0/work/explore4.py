import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']; act = S['act']
m = (ctrl==0)&(ep==2)&(t==4000)
x = X[m][0]
print(tools.block(x))
print('logits', tools.forward(w, x))
# Phase served lanes (stop-line indices)
serve = {0:[0,1,6,7], 1:[2,8], 2:[3,4,9,10], 3:[5,11]}
serve_up = {0:[24,25,28,29], 1:[25,29], 2:[26,27,30,31], 3:[27,31]}
# Define "own empty, other waiting" states: ok==1, current phase lanes queue==0 & density==0, some other phase has queue>0
ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5
g = tools.greedy(w, X)
def own_load(x,p): return x[serve[p]].sum()+x[[i+12 for i in serve[p]]].sum()
own = np.array([own_load(X[i],ph[i]) for i in range(len(X))])
others = np.array([max(X[i][serve[q]].sum() for q in range(4) if q!=ph[i]) for i in range(len(X))])
sel = ok & (own==0) & (others>0)
for c in range(3):
    mc = sel&(ctrl==c)
    print(S['controller_names'][c], 'n states own-empty-others-waiting (ok):', mc.sum(), ' net holds:', (g[mc]==ph[mc]).sum())
# which states does net hold?
hold = sel & (g==ph)
print('hold breakdown by phase', [ (hold&(ph==p)).sum() for p in range(4)])
print('hold by time>3600', (hold & (t>=3600)).sum(), (hold&(t<3600)).sum())
# look at unique held states
idx = np.where(hold & (ctrl!=0) )[0][:20]
for i in idx:
    x=X[i]; print(S['controller_names'][ctrl[i]], ep[i], t[i], 'ph',ph[i],'tg',round(x[45]*60), 'q',x[0:12],'d', x[12:24], 'du', x[32:40], 'lg', tools.forward(w,x)[0])
