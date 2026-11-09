from common import *
np.set_printoptions(linewidth=220, precision=3, suppress=True)
m = OK & (CTRL == 0)
print("distinct values of q and d (times 3) in network-run ok states")
for i, n in enumerate(LANES):
    qv, qc = np.unique(np.round(X[m, i]*3, 2), return_counts=True)
    dv, dc = np.unique(np.round(X[m, 12+i]*3, 2), return_counts=True)
    print(n, "q:", dict(zip(qv, qc)), " d:", dict(zip(dv, dc)))
print("upstream q*34 / d*34 quantiles (ok, ctrl0)")
for k in range(24, 40):
    print(tools.names()[k], np.quantile(X[m, k]*34.13, [0.5, 0.9, 0.99, 1.0]).round(1))
