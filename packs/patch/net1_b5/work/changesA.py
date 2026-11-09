from common import *
from repairA import copy_cols
wA=copy_cols(W0,[('q_N1','q_S1'),('d_N1','d_S1'),('q_W0','q_E0')],biascomp=False)
for c in range(3):
    m=(ctl==c)&ok
    g0=tools.greedy(W0,X[m]); g1=tools.greedy(wA,X[m])
    T=np.zeros((4,4),int)
    for a,b in zip(g0[g0!=g1],g1[g0!=g1]): T[a,b]+=1
    L=Lq[m]; Ldm=Ld[m]
    ch=g0!=g1
    print(S['controller_names'][c],'changed',ch.sum(),'\n',T)
    print('   halting load of picked phase: old',L[ch][np.arange(ch.sum()),g0[ch]].mean().round(2),'new',L[ch][np.arange(ch.sum()),g1[ch]].mean().round(2),
          '  density load old',Ldm[ch][np.arange(ch.sum()),g0[ch]].mean().round(2),'new',Ldm[ch][np.arange(ch.sum()),g1[ch]].mean().round(2))
    # N-lanes vs S lanes in changed states
    print('   in changed states mean q_N1,q_S1,q_W0,q_E0', X[m][ch][:,[ix['q_N1'],ix['q_S1'],ix['q_W0'],ix['q_E0']]].mean(0))
tools.save(wA,'/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wA.npz')
print(tools.edit_size(tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wA.npz'),W0))
print(tools.trainability(wA))
