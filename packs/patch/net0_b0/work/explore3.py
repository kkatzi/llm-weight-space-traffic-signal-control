import sys; sys.path.insert(0, '.')
import numpy as np, tools
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']; act = S['act']
names = tools.names()
m = (ctrl==0)&(ep==2)
Xe=X[m]; te=t[m]; ae=act[m]
print('first 30 decisions:')
for i in range(30):
    x=Xe[i]; print(te[i], 'ph',x[40:44].argmax(),'ok',int(x[44]),'tg',round(x[45]*60,1),'act',ae[i], 'q',x[0:12], 'qu', x[24:32])
print('after 3600:')
idx = np.where(te>=3500)[0]
for i in idx[::10][:60]:
    x=Xe[i]; print(te[i], 'ph',x[40:44].argmax(),'ok',int(x[44]),'tg',round(x[45]*60,1),'act',ae[i], 'q',x[0:12],'d',x[12:24], 'qu', x[24:32],'du',x[32:40])
