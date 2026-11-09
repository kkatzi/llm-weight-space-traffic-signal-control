import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; act = S['act']; t = S['time_s']
names = tools.names()
mm = (ctrl==0)&(ep==2)
Xe = X[mm]; te=t[mm]; ae=act[mm]
lg = tools.forward(w, Xe)
idx = np.where(te>=3500)[0]
for i in idx[::1][:80]:
    nz = {names[j]:round(Xe[i,j],2) for j in range(40) if Xe[i,j]>0}
    print(te[i], 'ph', Xe[i,40:44].argmax(), 'ok', Xe[i,44], 'tg %.2f'%Xe[i,45], 'act', ae[i], lg[i].round(1), nz)
