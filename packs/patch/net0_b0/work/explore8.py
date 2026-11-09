from common import *
np.set_printoptions(precision=2, suppress=True, linewidth=250)
w = tools.load("../start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']; act=S['act']
def show(e, l, c=0, pad=3):
    me = (ctrl==c)&(ep==e); Xe = X[me]; te = t[me]; ae = act[me]; phe = Xe[:,40:44].argmax(1)
    p = lane_phase(l); run=0; best=0; bi=0
    for i in range(len(Xe)):
        if Xe[i,l]>0 and phe[i]!=p: run+=1
        else: run=0
        if run>best: best=run; bi=i
    for i in range(bi-best-pad, bi+pad):
        x=Xe[i]; print(te[i],'ph',phe[i],'tg',round(x[45]*60),'a',ae[i],'q',x[0:12],'qu',x[24:32],'du',x[32:40],'lg',tools.forward(w,x)[0])
show(1, 6)
print()
show(5, 8)
