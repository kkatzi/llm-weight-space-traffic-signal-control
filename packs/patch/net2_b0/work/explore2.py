import numpy as np, sys
sys.path.insert(0, '.')
import tools
np.set_printoptions(linewidth=200, precision=3, suppress=True)
w = tools.load("start_weights.npz")
S = tools.states()
X = S["obs"].astype(np.float64); act = S["act"]; ctl = S["controller"]; ep = S["episode"]; t = S["time_s"]
names = tools.names()
# phase lanes
# stop-line lanes per phase: phase0 NS through: N0,N1,S0,S1 -> idx 0,1,6,7 ; phase1 NS left: N2,S2 -> 2,8
# phase2 EW through: E0,E1,W0,W1 -> 3,4,9,10; phase3 EW left: E2,W2 -> 5,11
PL = {0:[0,1,6,7],1:[2,8],2:[3,4,9,10],3:[5,11]}
for c in range(3):
    m = ctl==c
    print("controller", S["controller_names"][c])
    for e in range(6):
        me = m & (ep==e)
        tt = t[me]
        print(f"  ep{e}: n={me.sum()}, tmax={tt.max()}")
# per controller episode length -> rough TTS proxy: sum of densities?
# Network episodes: phase sequence and green durations
m = ctl==0
for e in range(2):
    me = np.where(m & (ep==e))[0]
    ph = X[me,40:44].argmax(1)
    tg = X[me,45]
    a = act[me]
    # list green runs
    runs=[]; start=0
    for i in range(1,len(ph)+1):
        if i==len(ph) or ph[i]!=ph[start]:
            runs.append((ph[start], t[me][start], (i-start)))
            start=i
    print("ep",e,"num runs",len(runs))
    print([ (r[0], r[2]*5) for r in runs[:80]])
