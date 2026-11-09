from common import *
serve={0:[0],1:[0],2:[1],3:[2],4:[2],5:[3],6:[0],7:[0],8:[1],9:[2],10:[2],11:[3]}
lane_names=[n[2:] for n in N[0:12]]
def synth(lane, phase, tg, nveh=1, halting=True):
    x=np.zeros(46)
    v=nveh/cs
    x[12+lane]=min(v,1)
    if halting: x[lane]=min(v,1)
    x[40+phase]=1; x[44]=1.0 if tg>=10 else 0.0; x[45]=min(tg/60,1)
    return x
def check(w,label):
    bad=[]
    for lane in range(12):
        for p in range(4):
            if p in serve[lane]: continue
            for nveh in (1,2,3):
                for tg in (12,17,22,32,47,60):
                    for halting in (True,False):
                        x=synth(lane,p,tg,nveh,halting)
                        a=tools.greedy(w,x)[0]
                        if a==p: bad.append((lane_names[lane],p,nveh,tg,halting,a))
    print(label,'stuck cases (net stays on a phase not serving the only vehicle):',len(bad))
    for b in bad[:40]: print('   ',b)
    return bad
if __name__=='__main__':
    import sys
    check(W0,'start')
    check(tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wA.npz'),'wA')
    # empty network: what does it pick
    for w,l in [(W0,'start'),(tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wA.npz'),'wA')]:
        for p in range(4):
            x=np.zeros(46); x[40+p]=1; x[44]=1; x[45]=0.2
            print(l,'empty, phase',p,'->',tools.greedy(w,x)[0], tools.forward(w,x)[0].round(2))
