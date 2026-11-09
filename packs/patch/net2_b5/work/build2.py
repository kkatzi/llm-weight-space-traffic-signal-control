"""Hand construction of the edited network (no fitting).

Rules (each implemented by one repurposed layer-1 'detector' unit and a layer-2 'gate' unit that pushes one logit
down by M when it fires):
  R1  phase 3 (EW_left) is green, ok=1, both EW left pockets empty (density 0), at least one vehicle halting on any
      other stop-line lane                                   -> do not pick phase 3 (no idle hold)
  E_p for p=0..3: ok=1, no MOVING vehicle on any upstream lane (upstream density == upstream queue on all 8 lanes;
      in practice only true in the final drain or the very first seconds),
      the stop-line lanes of phase p are empty (density 0)   -> do not pick phase p
      (so in the drain only phases that actually have vehicles at the stop line can get green; when every phase is
      empty all four logits drop by the same amount and the original choice is kept)

Layer-1 detectors: tanh units saturated at +1 or -1 except when firing.  Their old outgoing weights are kept; the
mean shift that replacing the old unit by a constant causes is moved into b2 (one b2 entry per layer-2 unit, shared
by all detectors).  R1's detector (unit 50) additionally has its outgoing column zeroed except for its gate, so the
choice among the remaining phases when R1 fires is the original network's.
Layer-2 gates: existing low-importance units; a large weight L from the detector saturates them at -1 normally and
+1 when the detector fires; their W3 entry for the target logit is set to -M/2 and b3 cancels the +M/2 offset and
the mean contribution of the unit's old output.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tools, numpy as np

SERVE = {0: [0, 1, 6, 7], 1: [2, 8], 2: [3, 4, 9, 10], 3: [5, 11]}
OK, PH0 = 44, 40
UPD = list(range(32, 40))           # upstream densities
UPQ = list(range(24, 32))           # upstream queues (halting), same lane order
H1_UNITS = {'R1': 50, 'E0': 42, 'E1': 38, 'E2': 5, 'E3': 48}
GATES = {3: 1, 0: 63, 1: 20, 2: 27}     # target logit -> layer-2 gate unit
RULE_TARGET = {'R1': 3, 'E0': 0, 'E1': 1, 'E2': 2, 'E3': 3}
L, M = 12.0, 30.0


def detector_rows():
    """returns {rule: (row46, bias, fire_sign)}; pre-activation > 0 means 'fires'."""
    rows = {}
    # R1 (full row): A*(ph3 + ok - 2) - K*(d_E2 + d_W2) + c*sum(q others) + beta
    r = np.zeros(46); A, K, c, beta = 80.0, 200.0, 6.0, -1.0
    others = [l for l in range(12) if l not in SERVE[3]]
    r[others] = c; r[[12 + l for l in SERVE[3]]] = -K; r[PH0 + 3] = A; r[OK] = A
    rows['R1'] = (r, -2 * A + beta)
    # E_p (sparse rows): A*(ok - 1) - K*sum(d of p's lanes) - U*sum(upstream density - upstream queue) + beta
    for p in range(4):
        r = {}
        A, K, U, beta = 30.0, 60.0, 600.0, 8.0
        r[OK] = A
        for l in SERVE[p]: r[12 + l] = -K
        for j in UPD: r[j] = -U
        for j in UPQ: r[j] = +U
        rows['E%d' % p] = (r, -A + beta)
    return rows


def build(w0, X, e_full_rows=False, use=('R1', 'E0', 'E1', 'E2', 'E3')):
    w = {k: v.copy() for k, v in w0.items()}
    h1o, h2o = tools.hidden(w0, X)
    rows = detector_rows()
    normal = {}
    for name in use:
        j = H1_UNITS[name]; row, bias = rows[name]
        if isinstance(row, dict):
            if e_full_rows:
                w['W1'][j] = 0.0
            for k, v in row.items(): w['W1'][j, k] = v
        else:
            w['W1'][j] = row
        w['b1'][j] = bias
        normal[name] = -1.0        # detectors sit at -1 unless firing
    # layer 2: column handling + b2 compensation (shared)
    b2 = w0['b2'].copy()
    for name in use:
        j = H1_UNITS[name]
        m_old = h1o[:, j].mean()
        if name == 'R1':
            for k in range(64):
                if k == GATES[RULE_TARGET[name]]: continue
                b2[k] += w0['W2'][k, j] * m_old       # remove old mean contribution, column -> 0
                w['W2'][k, j] = 0.0
        else:
            for k in range(64):
                if k in GATES.values(): continue
                b2[k] += w0['W2'][k, j] * (m_old - normal[name])
    w['b2'] = b2
    # gates
    gate_rules = {}
    for name in use: gate_rules.setdefault(GATES[RULE_TARGET[name]], []).append(name)
    h1n = np.tanh(X @ w['W1'].T + w['b1'])
    b3 = w0['b3'].copy()
    for i, names in gate_rules.items():
        for name in names: w['W2'][i, H1_UNITS[name]] = L
        det = [H1_UNITS[n] for n in names]
        pre_wo = h1n @ w['W2'][i] - h1n[:, det] @ w['W2'][i, det] + w0['b2'][i]
        centre = 0.5 * (pre_wo.min() + pre_wo.max())
        # normal: all detectors -1 -> pre = pre_wo - L*n ; one fires -> pre_wo - L*(n-2). Put threshold between.
        n = len(names)
        w['b2'][i] = w0['b2'][i] - centre + L * (n - 1)
        p = [RULE_TARGET[nm] for nm in names][0]
        m2 = h2o[:, i].mean()
        old_col = w0['W3'][:, i].copy()
        w['W3'][p, i] = -M / 2
        # normal gate output -1: new contribution = -W3[:,i]; old mean contribution = old_col*m2
        b3 += old_col * m2 + w['W3'][:, i]
        b3[p] -= 0.0
    w['b3'] = b3
    return w


if __name__ == '__main__':
    w0 = tools.load('start_weights.npz')
    X = tools.states()['obs'].astype(float)
    full = '--full' in sys.argv
    out = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = out[0] if out else 'work/r2.npz'
    w = build(w0, X, e_full_rows=full)
    print(tools.edit_size(w, w0))
    tools.save(w, out); print('saved', out)
