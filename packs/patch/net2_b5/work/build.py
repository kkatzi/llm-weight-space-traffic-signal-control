"""Construct the edited network by hand (no fitting).

Rule R1 ("no idle EW-left hold"): if the current green is phase 3 (EW_left), the minimum green is over (ok=1),
both EW left pockets (E2, W2) are empty (density 0) and at least one vehicle halts on any other stop-line lane,
then phase 3 must not be picked again: switch to whichever other phase the original network prefers.

Implementation:
  h1 unit 50 (least important layer-1 unit) is rewritten as a detector: ~ -1 normally, ~ +1 when R1 fires.
  Its old outgoing weights W2[:,50] are zeroed (option) and their mean effect moved into b2.
  h2 unit 1 (least important layer-2 unit) becomes a gate: tanh(old_input + L*h1_50 + offset) -> ~-1 normally,
  ~+1 when R1 fires.  Its W3 column becomes [0,0,0,-M/2]; b3 absorbs the old mean contribution and +M/2.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np

J, I = 50, 1          # repurposed h1 unit, h2 unit
OTHERS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]     # halting inputs of the stop-line lanes not served by phase 3
D_E2, D_W2 = 17, 23
PH3, OK = 43, 44


def build(w0, zero_col=True, c=6.0, beta=-1.0, K=200.0, A=80.0, L=10.0, M=12.0, X=None):
    w = {k: v.copy() for k, v in w0.items()}
    h1, h2 = tools.hidden(w0, X)
    m1, m2 = h1[:, J].mean(), h2[:, I].mean()
    # ---- layer 1: detector unit J
    row = np.zeros(46)
    row[OTHERS] = c          # each halting vehicle elsewhere adds c/2.987
    row[[D_E2, D_W2]] = -K   # any vehicle in an EW left pocket kills it
    row[[PH3, OK]] = A       # needs phase 3 and ok
    w['W1'][J] = row
    w['b1'][J] = -2 * A + beta
    # ---- layer 2: remove unit J's old influence on all other units
    for k in range(64):
        if k == I:
            continue
        if zero_col:
            w['b2'][k] = w0['b2'][k] + w0['W2'][k, J] * m1
            w['W2'][k, J] = 0.0
        else:
            w['b2'][k] = w0['b2'][k] + w0['W2'][k, J] * (m1 + 1.0)   # new unit sits at -1
    # gate unit I: keep its other inputs, centre them, add a large weight from J
    pre_wo = h1 @ w0['W2'][I] - h1[:, J] * w0['W2'][I, J] + w0['b2'][I]
    centre = 0.5 * (pre_wo.min() + pre_wo.max())
    w['W2'][I, J] = L
    w['b2'][I] = w0['b2'][I] - centre
    # ---- output layer
    new_col = np.array([0.0, 0.0, 0.0, -M / 2])
    w['W3'][:, I] = new_col
    w['b3'] = w0['b3'] + w0['W3'][:, I] * m2 + new_col   # old mean contribution kept, gate offset cancelled
    return w


if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    X = tools.states()['obs'].astype(float)
    zero_col = '--keepcol' not in sys.argv
    w = build(w0, zero_col=zero_col, X=X)
    print(tools.edit_size(w, w0))
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'work/r1.npz'
    tools.save(w, out)
    print('saved', out)
