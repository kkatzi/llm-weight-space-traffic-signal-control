from common import *
w = tools.load("../start_weights.npz")
rng = np.random.default_rng(0)
def rand_sparse(n):
    X = np.zeros((n, 46))
    for k in range(n):
        nl = rng.integers(0, 4)
        for l in rng.choice(12, nl, replace=False):
            c = rng.integers(1, 4); X[k, l] = X[k, 12 + l] = min(1, c * ONE)
            if rng.random() < 0.3: X[k, l] = max(0, X[k, l] - ONE)  # some moving
        nu = rng.integers(0, 4)
        for u in rng.choice(8, nu, replace=False):
            c = rng.integers(1, 4); X[k, 32 + u] = c * ONEU
            if rng.random() < 0.3: X[k, 24 + u] = X[k, 32 + u]
        p = rng.integers(4); X[k, 40 + p] = 1
        tg = rng.choice([12, 17, 22, 27, 32, 42, 60]); X[k, 44] = 1; X[k, 45] = tg / 60
    return X
X = rand_sparse(200000)
E = phase_empty(X)
anyload = (~E).any(1)
for vs in [(), (1, 2, 3), (0, 1, 2, 3)]:
    a = make_policy(w, vs)(X)
    bad = anyload & E[np.arange(len(X)), a]
    print('veto', vs, 'picks empty phase while some loaded:', bad.sum(), 'by action', np.bincount(a[bad], minlength=4))
