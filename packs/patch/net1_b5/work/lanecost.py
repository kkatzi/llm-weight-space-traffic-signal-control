from common import *
lane_names=[n[2:] for n in N[0:12]]
for c in range(3):
    m=ctl==c
    # per-episode total halting and vehicles (proxy)
    tot_h=[(q[m&(ep==e)].sum()+qu[m&(ep==e)].sum())*5/3600 for e in range(6)]
    tot_v=[(d[m&(ep==e)].sum()+du[m&(ep==e)].sum())*5/3600 for e in range(6)]
    print(f'{S["controller_names"][c]:14s} halting veh-h per ep',np.round(tot_h,2),'sum',round(sum(tot_h),2),' vehicles-on-approach veh-h',np.round(tot_v,1), round(sum(tot_v),1))
print('per-lane halting veh-h summed over 6 episodes (stop-line)')
for c in range(3):
    m=ctl==c
    print(f'{S["controller_names"][c]:14s}', np.round(q[m].sum(0)*5/3600,2), ' up', np.round(qu[m].sum(0)*5/3600,2))
print(lane_names)
# per phase grouping
