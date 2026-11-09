"""Final patch.
 (1) W2 imputation: the dead q_W2/d_W2 inputs are read from q_E2/d_E2 (same demand, same phase 3):
       W1[:,5] += W1[:,11];  W1[:,17] += W1[:,23]                                        (128 params)
 (2) Stall breaker: layer-1 unit J (least used unit, replaced by its mean) becomes a detector
       h1_J = tanh(K*(t_green + phase2 - 1.975))  = +1 only when phase 2 has been held >= 60 s, else -1
     Its layer-2 column is set to the layer-2 shift that the ORIGINAL network would see, in the idle
     phase-2 state, if a W-left car (vW on q_W2,d_W2) and extra N/S through demand (vN on q,d of N0,N1,S0,S1)
     were present; b2 absorbs the old unit's mean and the detector's OFF level.
       W1[J,:] (44 live entries, 2 shared with (1)), b1[J], W2[:,J] (64), b2 (64)
"""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools
NS_THRU = [0, 1, 6, 7, 12, 13, 18, 19]
DEAD = [11, 23]

def build(vW=0.67, vN=0.25, J=57, K=200.0, thr=1.975, w0=None):
    w0 = tools.load('start_weights.npz') if w0 is None else w0
    S = tools.states(); X = S['obs'].astype(np.float64)
    w = {k: v.copy() for k, v in w0.items()}
    W1 = w0['W1']
    # (1) imputation
    w['W1'][:, 5] += W1[:, 11]
    w['W1'][:, 17] += W1[:, 23]
    # mean of unit J under the imputed network, on the start network's own recorded states
    h1, _ = tools.hidden(w, X[S['controller'] == 0])
    mu = h1[:, J].mean()
    # layer-2 shift the original network would get from the imputed idle state
    xe = np.zeros(46); xe[42] = 1; xe[44] = 1; xe[45] = 1.0
    xi = xe.copy(); xi[DEAD] = vW; xi[NS_THRU] += vN
    dh1 = np.tanh(W1 @ xi + w0['b1']) - np.tanh(W1 @ xe + w0['b1'])
    dh1[J] = 0.0
    shift = w0['W2'] @ dh1
    on = np.tanh(K * (1.0 + 1.0 - thr))            # detector level when ON (= -OFF level)
    # (2) detector unit
    live = [i for i in range(46) if i not in DEAD]
    w['W1'][J, live] = 0.0
    w['W1'][J, 45] = K; w['W1'][J, 42] = K
    w['b1'][J] = -K * thr
    newcol = shift / (2 * on)
    w['b2'] = w0['b2'] + w0['W2'][:, J] * mu + newcol * on
    w['W2'][:, J] = newcol
    return w

if __name__ == '__main__':
    vW, vN, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    w = build(vW, vN)
    tools.save(w, out)
    print(out, tools.edit_size(tools.load(out), tools.load('start_weights.npz')))
