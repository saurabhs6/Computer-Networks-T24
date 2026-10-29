"""Check z10_policy (Theorem 1 of [10]) against a direct numerical solution of problem P1."""
import numpy as np
from scipy.optimize import minimize
import ota_fl_sim as S
rng = np.random.default_rng(0)
worst = -np.inf; rel = []
for trial in range(40):
    K = 12
    C = np.abs(rng.normal(size=K) + 1j * rng.normal(size=K)) * rng.uniform(0.5, 2)
    beta = 10 ** rng.uniform(-1, 3); N0 = 10 ** rng.uniform(-3, 1)
    obj = lambda G, x: beta / (beta + 1) * np.sum((G - 1) ** 2) + (G.sum() - K) ** 2 / (beta + 1) + N0 * x * x
    G, sq = S.z10_policy(C, beta, N0)
    f_th = obj(G, 1 / sq)
    assert np.all(G <= C / sq + 1e-9) and np.all(G >= -1e-12)
    best = np.inf
    for x0 in np.geomspace(0.05, 20, 25):          # direct search: for fixed x the problem in G is a convex QP with box constraints
        def inner(x):
            res = minimize(lambda g: obj(g, x), np.minimum(1.0, C * x), bounds=[(0, c * x) for c in C], method="L-BFGS-B")
            return res.fun
        r = minimize(lambda lx: inner(np.exp(lx[0])), [np.log(x0)], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10))
        best = min(best, r.fun)
    worst = max(worst, (f_th - best) / max(best, 1e-12)); rel.append((f_th - best) / max(best, 1e-12))
print("relative difference (Theorem 1 - direct), max %.2e min %.2e" % (max(rel), min(rel)))
