"""Build edited weights by construction.

Knob A (alpha): scale how strongly the network perceives the four left-turn pockets.
  W1[j, col] *= alpha for col in the left-pocket queue/density inputs (q_N2,q_E2,q_S2,q_W2,d_N2,d_E2,d_S2,d_W2)
  restricted to the TOPK hidden units j with the largest |W1[j, col]| per column (fixed set of positions).
"""
import numpy as np, sys, copy, json
sys.path.insert(0, '.')
import tools

W0 = tools.load('start_weights.npz')
LEFT_Q = [2, 5, 8, 11]; LEFT_D = [14, 17, 20, 23]

def left_positions(topk=32):
    pos = []
    for col in LEFT_Q + LEFT_D:
        idx = np.argsort(-np.abs(W0['W1'][:, col]))[:topk]
        pos += [(int(j), col) for j in idx]
    return pos

def build(alpha=1.0, topk=32, b3_delta=None):
    w = copy.deepcopy(W0)
    for j, col in left_positions(topk):
        w['W1'][j, col] = W0['W1'][j, col] * alpha
    if b3_delta is not None:
        w['b3'] = W0['b3'] + np.asarray(b3_delta, dtype=float)
    return w

if __name__ == '__main__':
    name = sys.argv[1]; alpha = float(sys.argv[2]); topk = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    w = build(alpha, topk)
    out = f'work/{name}.npz'
    tools.save(w, out)
    w_back = tools.load(out)
    print(out, tools.edit_size(w_back, W0))
    S = tools.states(); X = S['obs'].astype(np.float64)
    ok = X[:, 44] > 0.5; m = ok & (S['controller'] == 0)
    print('ok-decision change rate on start-net states:', float((tools.greedy(w_back, X)[m] != tools.greedy(W0, X)[m]).mean()))
