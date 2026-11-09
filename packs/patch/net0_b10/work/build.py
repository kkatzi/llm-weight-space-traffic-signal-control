"""Build the edited network: 'empty-phase gates'.

For every phase p one layer-1 unit (K1[p]) is rewritten to detect "no vehicle on any lane served by phase p"
(stop-line lanes + the upstream lanes that feed them). One layer-2 unit (K2[p]) gets a dominating weight from
that gate, so it sits at -1 normally and +1 when the gate fires. Its output column in W3 is replaced by a pure
penalty on logit p. Net effect: a phase whose lanes are all empty loses PEN logit units, so the network never
selects (or holds) an empty phase while another phase has vehicles. When all phases are empty, all four logits
drop by the same amount and the original choice is kept.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tools

N = tools.names(); ix = {n: i for i, n in enumerate(N)}
SL = {0: ['N0', 'N1', 'S0', 'S1'], 1: ['N2', 'S2'], 2: ['E0', 'E1', 'W0', 'W1'], 3: ['E2', 'W2']}
UP = {0: ['N0u', 'N1u', 'S0u', 'S1u'], 1: ['N1u', 'S1u'], 2: ['E0u', 'E1u', 'W0u', 'W1u'], 3: ['E1u', 'W1u']}

# units chosen because replacing them by a constant changes the fewest recorded decisions (analyze_units.py)
K1 = {0: 32, 1: 60, 2: 57, 3: 59}         # layer-1 gate units
C1 = {32: +1, 60: +1, 57: +1, 59: -1}     # value each unit takes in normal (non-empty) states
K2 = {0: 37, 1: 0, 2: 5, 3: 49}          # layer-2 relay units (choose_units.py)

B = 4.0            # gate bias: empty lanes -> |pre-act| = 4 -> |tanh| = 0.9993
C_SL = 36.0        # per stop-line density unit (1 vehicle = 0.335 -> 12)
C_UP = 410.0       # per upstream density unit (1 vehicle = 0.0293 -> 12)
V = 20.0           # relay weight (layer-2 rows have |pre-act| <= 13.7 otherwise)
MOVING_ONLY = '--allup' not in sys.argv   # halted upstream vehicles are blocked behind a full stop-line lane
PEN = 20.0         # logit penalty for an empty phase (max logit spread on recorded states 14.8)


def build(w0, comp_b2=False, comp_b3=True, S=None):
    w = {k: v.copy() for k, v in w0.items()}
    if S is not None:
        h1, h2 = tools.hidden(w0, S)
    for p in range(4):
        k, j, c = K1[p], K2[p], C1[K1[p]]
        # layer 1: g = tanh(s * (B - demand)), s = -c  -> normal value c, fired value -c
        s = -c
        w['W1'][k, :] = 0.0
        for l in SL[p]:
            w['W1'][k, ix['d_' + l]] = -s * C_SL
        for u in UP[p]:
            w['W1'][k, ix['d_' + u]] = -s * C_UP        # moving upstream vehicles only: C_UP * (d - q)
            if MOVING_ONLY:
                w['W1'][k, ix['q_' + u]] = +s * C_UP
        w['b1'][k] = s * B
        # layer 2: relay j. Normal: g=c -> pre += V*(-1) -> h2 = -1; fired: g=-c -> +V -> h2 = +1
        w['W2'][j, k] = -c * V
        # layer 3: replace column j by a penalty on logit p
        if comp_b3 and S is not None:
            w['b3'] += w0['W3'][:, j] * h2[:, j].mean() - 0.0
        w['W3'][:, j] = 0.0
        w['W3'][p, j] = -PEN / 2      # normal h2=-1: +PEN/2 on every logit (uniform, cancels); fired: -PEN/2
    if comp_b2 and S is not None:
        relays = set(K2.values())
        for p in range(4):
            k, c = K1[p], C1[K1[p]]
            for i in range(64):
                if i in relays:
                    continue
                w['b2'][i] += w0['W2'][i, k] * (h1[:, k].mean() - c)
    return w


if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    X = tools.states()['obs'].astype(np.float64)
    comp_b2 = '--b2' in sys.argv
    w = build(w0, comp_b2=comp_b2, S=X)
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'work/weights.npz'
    tools.save(w, out)
    print('saved', out, tools.edit_size(tools.load(out), w0))
