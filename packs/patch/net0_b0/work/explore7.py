from common import *
np.set_printoptions(precision=3, suppress=True, linewidth=200)
w = tools.load("../start_weights.npz")
S = tools.states()
X = S['obs'].astype(np.float64); ctrl = S['controller']; ep = S['episode']; t = S['time_s']
# unique density values
print('unique stop-line d values', np.unique(np.round(X[:,12:24],4))[:10])
print('unique upstream d values', np.unique(np.round(X[:,32:40],4))[:10])
print('unique upstream q values', np.unique(np.round(X[:,24:32],4))[:10])
# phase loads
DL = {0:[12,13,18,19,32,33,36,37], 1:[14,20,33,37], 2:[15,16,21,22,34,35,38,39], 3:[17,23,35,39]}
load = np.stack([X[:,DL[q]].sum(1) for q in range(4)],1)
empty = load < 0.5*ONEU
g = tools.greedy(w, X); ph = X[:,40:44].argmax(1); ok = X[:,44]>0.5
anyload = (~empty).any(1)
for c in range(3):
    m = (ctrl==c)&ok
    bad = m & empty[np.arange(len(X)), g] & anyload
    print(S['controller_names'][c], 'ok states', m.sum(), 'net picks empty phase while others loaded:', bad.sum(),
          ' of which hold:', (bad&(g==ph)).sum(), ' t<3600:', (bad&(t<3600)).sum())
    print('   share of states with phase q empty', empty[m].mean(0).round(3), ' all empty', empty[m].all(1).mean().round(3))
# t_green distribution for net
m = ctrl==0
print('t_green==1 share (net, ok states, t<3600)', ((X[:,45]>=0.999)&m&ok&(t<3600)).sum())
# max consecutive waiting (red with queue) per lane, by controller
for c in range(3):
    mx = []
    for e in range(6):
        me = (ctrl==c)&(ep==e)
        Xe = X[me]; phe = Xe[:,40:44].argmax(1)
        for l in range(12):
            p = lane_phase(l); run = 0; best = 0
            for i in range(len(Xe)):
                if Xe[i,l] > 0 and phe[i] != p: run += 5; best = max(best, run)
                else: run = 0
            mx.append((best, e, l))
    mx.sort(reverse=True)
    print(S['controller_names'][c], 'longest red-with-queue runs (s, ep, lane):', mx[:8])
