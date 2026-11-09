from common import *
# phase -> (stop-line density input indices, upstream density input indices)
STOP={0:['d_N0','d_N1','d_S0','d_S1'],1:['d_N2','d_S2'],2:['d_E0','d_E1','d_W0','d_W1'],3:['d_E2','d_W2']}
UP={0:['d_N0u','d_N1u','d_S0u','d_S1u'],1:['d_N1u','d_S1u'],2:['d_E0u','d_E1u','d_W0u','d_W1u'],3:['d_E1u','d_W1u']}
def counts(Xa):
    V=np.zeros((len(Xa),4))
    for p in range(4):
        V[:,p]=sum(Xa[:,ix[n]]*cs for n in STOP[p])+sum(Xa[:,ix[n]]*cu for n in UP[p])
    return V
def build(w0, G=(31,0,59,54), R=(25,35,39,15), K=10.0, margin=4.0, relayW=40.0, Delta=30.0, b2comp=True, Xref=None, verbose=False):
    w={k:v.copy() for k,v in w0.items()}
    Xref = X[ctl==0] if Xref is None else Xref
    H1old=np.tanh(Xref@w0['W1'].T+w0['b1']); H2old=np.tanh(H1old@w0['W2'].T+w0['b2'])
    for p in range(4):
        i=G[p]
        gate_cols=[ix[n] for n in STOP[p]+UP[p]]
        rest=[c for c in range(46) if c not in gate_cols]
        bound=np.abs(w0['W1'][i,rest]).sum()       # worst case |old_rest . x| for x in [0,1]
        # pre = B + old_rest.x - K' * S_p ;  K' per vehicle
        B=bound+margin
        Kp=2*bound+2*margin    # one vehicle drives pre to <= -margin
        for n in STOP[p]: w['W1'][i,ix[n]]=-Kp*cs
        for n in UP[p]: w['W1'][i,ix[n]]=-Kp*cu
        w['b1'][i]=B
        if verbose: print(f'phase {p}: L1 unit {i} rest-bound {bound:.2f} B {B:.2f} K/veh {Kp:.2f}')
    H1new=np.tanh(Xref@w['W1'].T+w['b1'])
    if b2comp:
        # keep mean pre-activation of every layer-2 unit (other than relays) unchanged
        dmean=(H1old[:,list(G)]-H1new[:,list(G)]).mean(0)
        for j in range(64):
            if j in R: continue
            w['b2'][j]+= w0['W2'][j,list(G)]@dmean
    b3=w['b3'].copy()
    for p in range(4):
        r=R[p]
        w['W2'][r,G[p]]=relayW
        # old mean contribution of relay r moves into b3
        b3+= w0['W3'][:,r]*H2old[:,r].mean()
        w['W3'][:,r]=0.0
        w['W3'][p,r]=-Delta/2
        b3[p]-= Delta/2           # relay = -1 when gate inactive -> contributes +Delta/2; cancel it
    w['b3']=b3
    return w
if __name__=='__main__':
    wg=build(W0,verbose=True)
    print(tools.edit_size(wg,W0))
