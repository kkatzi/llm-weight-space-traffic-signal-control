from common import *
from build_gate import counts
wG=tools.load('/home/traffic-ntua/wsp_v3_agent_runs/patch/net1_b5/work/wG.npz')
V=counts(X)
for c in range(3):
    m=np.where((ctl==c)&ok)[0]
    g=tools.greedy(wG,X[m]); Vm=V[m]
    vc=Vm[np.arange(len(m)),g]; vmax=Vm.max(1)
    print(S['controller_names'][c],'n',len(m))
    for a,b in [(0.5,1.5),(1.5,2.5),(2.5,4.5)]:
        mm=(vc>a)&(vc<b)
        print(f'   chosen V in ({a},{b}): {mm.sum()}  with max V>=6: {(mm&(vmax>=6)).sum()}  >=10: {(mm&(vmax>=10)).sum()}')
