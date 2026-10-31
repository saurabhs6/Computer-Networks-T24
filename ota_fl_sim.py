"""
T24 - Over-the-Air Federated Learning: Power Control Under Fading
Phases P2-P6: M1, M2, M3 simulation, joint evaluation and graphs.

Run:   python ota_fl_sim.py            (writes results.json + figures)
Needs: numpy, scikit-learn, matplotlib

Model of one communication round (matches Section 3.1 of the manuscript)
------------------------------------------------------------------------
* K devices hold non-IID shards of the scikit-learn digits data set.
* Each device computes a local gradient g_k of a softmax-regression model
  (d = 650 parameters) and sends the normalised symbol s_k = g_k / RHO.
* Device k scales by b_k = sqrt(eta) * conj(c_k_hat) / |c_k_hat|^2
  (channel inversion on its *estimated* effective channel c_k_hat).
  Per-device constraint:  |b_k|^2 * ||s_k||^2 / d <= P_MAX.
* Server receives y = sum_k c_k b_k s_k + n, n ~ CN(0, sigma^2),
  and forms  g_hat = RHO * Re(y) / (|S| * sqrt(eta)).
* SNR = P_MAX / sigma^2 (in dB).

M1 : single antenna, full participation, channel inversion (eta set by the
     weakest device so that every device meets its power limit).
M2 : single antenna, truncated channel inversion (devices with |c_hat| below
     a threshold stay silent; eta set by the weakest participant).
M3 : N_RX receive antennas; the receive combiner a is optimised jointly with
     the truncation rule (max-min effective gain), then truncated inversion.
M1-BF (ablation) : beamforming only, no truncation.
IDEAL : noiseless, fading-free aggregation (upper reference).

Baselines added for the second-level corrections (all re-implementations, see the manuscript):
M2-EIG : M2 with a fixed eigenvector combiner (no max-min refinement).
OPT    : threshold structure of the MSE-optimised policy of Cao et al. [7].
RCI    : regularised channel inversion with a tuned regulariser.
C22    : Case I power control of Cao et al. [12] ("C12" in the manuscript).
Z10    : closed-form gradient-statistics-aware power control of Zhang & Tao [10] (Theorem 1).
Y14    : statistics-based regularised policy with the closed-form structure of Yu et al. [14].
Y4     : device selection + receive beamforming of Yang et al. [4] (SDR ordering + DC rank-one
         feasibility detection), followed by channel inversion on the selected set.

Version 3 switches (defaults reproduce the earlier results exactly):
NORM      True -> unit-power symbols (each device divides by rho_k = ||g_k||/sqrt(d)).
JAKES_FD  normalised Doppler f_D*T of a first-order Gauss-Markov channel with Jakes
          correlation J0(2*pi*f_D*T); 0 keeps i.i.d. block fading.
N_RX      receive antennas of the multi-antenna methods.
DIRICHLET_ALPHA  label skew of the device partition.
"""
import json
from functools import lru_cache
import numpy as np
from sklearn.datasets import load_digits

# ------------------------------------------------------------------ settings
K = 50                 # devices (same as P1)
ROUNDS = 100           # training rounds per run (= P1 held-out channel block)
LR = 1.0               # server learning rate
C22_KAPPA = 1.0
NORM = False          # True: unit-power normalised symbols (review item 3)
RHO = 0.1              # gradient normalisation constant (pre-processing)
P_MAX = 1.0            # per-element power budget of the normalised symbol
N_RX = 4               # receive antennas for M3
TRAIN_SLICE = slice(0, 100)     # P1 train rounds used for tuning
TEST_SLICE = slice(400, 500)    # P1 held-out rounds used for reporting
SEEDS = list(range(10))
SNR_REF = -25          # reference operating SNR (dB): noise-limited regime where methods differ
P1_SEED = 42           # seed of the shared P1 channel trace
DIRICHLET_ALPHA = 0.5  # label skew across devices
JAKES_FD = 0.0         # 0: i.i.d. block fading; >0: Gauss-Markov fading with Jakes correlation
DEBIAS = False         # True: the server divides by the sum of the (estimated) received amplitudes instead of K*sqrt(eta)
SCALING = "mean"       # unit power: "mean" -> s_k = g_k/rho_k, server multiplies by mean(rho_k); "max" -> common s_k = g_k/rho_max
Y14_CAP = False        # True: Y14 also respects the per-round peak limit |b_k|^2 pi_k <= P_MAX
PILOT = False          # True (with DEBIAS): the amplitude sum is measured with one noisy pilot symbol through the true channel
PILOT_RNG = None
Y4_SOLVER = "CLARABEL"

# ---------------------------------------------------------------------- data
def load_data(seed):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(1000 + seed)
    idx = rng.permutation(len(X))
    n_test = 450
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def partition(y, k, alpha, rng):
    classes = np.unique(y)
    while True:
        parts = [[] for _ in range(k)]
        for c in classes:
            ids = rng.permutation(np.where(y == c)[0])
            p = rng.dirichlet(alpha * np.ones(k))
            cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
            for dev, chunk in enumerate(np.split(ids, cuts)):
                parts[dev].extend(chunk.tolist())
        if min(len(p) for p in parts) >= 5:
            return [np.array(p) for p in parts]


# --------------------------------------------------------------------- model
D_IN, N_CLS = 64, 10
D = D_IN * N_CLS + N_CLS


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def grad(w, X, y):
    W, b = w[:D_IN * N_CLS].reshape(D_IN, N_CLS), w[D_IN * N_CLS:]
    P = softmax(X @ W + b)
    P[np.arange(len(y)), y] -= 1.0
    P /= len(y)
    return np.concatenate([(X.T @ P).ravel(), P.sum(0)])


def accuracy(w, X, y):
    W, b = w[:D_IN * N_CLS].reshape(D_IN, N_CLS), w[D_IN * N_CLS:]
    return float(np.mean(np.argmax(X @ W + b, 1) == y))


# ------------------------------------------------------------------ channels
def cn(rng, shape, var=1.0):
    s = np.sqrt(var / 2.0)
    return rng.normal(0, s, shape) + 1j * rng.normal(0, s, shape)


def p1_channels(seed):
    """Single-antenna Rayleigh block, identical recipe to p1_channel_generation.py."""
    rng = np.random.default_rng(P1_SEED + seed)
    real = rng.normal(0.0, np.sqrt(0.5), size=(500, K))
    imag = rng.normal(0.0, np.sqrt(0.5), size=(500, K))
    return real + 1j * imag


def jakes_r(fd):
    """One-step correlation J0(2*pi*fd) of Clarke/Jakes fading."""
    from scipy.special import j0
    return float(j0(2 * np.pi * fd))


def gm_trace(rng, tail, fd, n=500):
    """First-order Gauss-Markov fading h[t] = r h[t-1] + sqrt(1-r^2) w[t] with r = J0(2 pi f_D T).
    Every h[t] is CN(0, 1); consecutive rounds are correlated."""
    r = jakes_r(fd)
    h = np.empty((n,) + tuple(tail), complex)
    h[0] = cn(rng, tail)
    for t in range(1, n):
        h[t] = r * h[t - 1] + np.sqrt(1 - r * r) * cn(rng, tail)
    return h


def channel_trace(method, seed):
    """Channel trace of one run: (500, K) for single-antenna methods, (500, K, N_RX) otherwise."""
    multi = method in ("M3", "M1-BF", "M2-EIG", "Y4")
    if JAKES_FD > 0:
        rng = np.random.default_rng(P1_SEED + seed + (177 if multi else 100))
        return gm_trace(rng, (K, N_RX) if multi else (K,), JAKES_FD)
    if multi:
        return cn(np.random.default_rng(P1_SEED + seed + 77), (500, K, N_RX), 1.0)
    return p1_channels(seed)


# ------------------------------------------------- [4]: device selection + beamforming
def y4_select(Hh, gamma, dc_iters=40, tol=1e-6):
    """Device selection and receive beamforming of Yang et al. [4] for the MSE requirement gamma
    (feasible device: ||m||^2 - gamma |m^H h_i|^2 <= 0, i.e. |a^H h_i|^2 >= 1/gamma for unit-norm a).
    Step 1 (sparsity inducing, PS1 of [4]):  min ||x||_1 - |||x|||_k + Tr(M) - ||M||_2
        s.t. Tr(M) - gamma h_i^H M h_i <= x_i, M >= 0, Tr(M) >= 1, x >= 0,
        solved by the DC algorithm (linearising the Ky Fan k-norm and the spectral norm); the smallest k
        with zero objective is found by bisection over k (the zero-objective condition is monotone in k).
    Step 2 (feasibility detection, PS2 of [4]): devices ordered by x in descending order; the largest
        nested set {pi(k),...,pi(M)} for which min Tr(M) - ||M||_2 reaches zero is found by bisection.
    Returns (mask of selected devices, unit-norm combiner a)."""
    import cvxpy as cp
    Kd, N = Hh.shape
    Q = np.stack([np.outer(h, h.conj()).T.flatten(order="F") for h in Hh])   # Re(Q_i . vec(M)) = h_i^H M h_i
    M = cp.Variable((N, N), hermitian=True)
    q = cp.real(Q @ cp.vec(M, order="F"))
    tr = cp.real(cp.trace(M))
    x = cp.Variable(Kd, nonneg=True)
    U = cp.Parameter((N, N), hermitian=True)
    wx = cp.Parameter(Kd, nonneg=True)
    p1 = cp.Problem(cp.Minimize(wx @ x + tr - cp.real(cp.trace(U @ M))),
                    [M >> 0, tr >= 1, tr - gamma * q <= x])
    msk = cp.Parameter(Kd, nonneg=True)
    p2 = cp.Problem(cp.Minimize(tr - cp.real(cp.trace(U @ M))),
                    [M >> 0, tr >= 1, cp.multiply(msk, tr - gamma * q) <= 0])

    def solve(p):
        try:
            p.solve(solver=Y4_SOLVER)
        except Exception:
            return False
        return p.status in ("optimal", "optimal_inaccurate") and M.value is not None

    def top(Mv):
        Mv = (Mv + Mv.conj().T) / 2
        w, V = np.linalg.eigh(Mv)
        return Mv, w, V[:, -1]

    def ps1(k):
        """DC algorithm for PS1 with Ky Fan k-norm; returns (zero objective?, x, a)."""
        U.value = np.zeros((N, N), complex); wx.value = np.ones(Kd)
        if not solve(p1):
            return False, None, None
        hist = []
        for _ in range(dc_iters):
            Mv, w, u = top(M.value); xv = np.maximum(x.value, 0)
            srt = np.sort(xv)[::-1]
            obj = srt[k:].sum() + (np.trace(Mv).real - w[-1])
            if obj < tol * max(1.0, np.trace(Mv).real):
                return True, xv, u
            hist.append(obj)
            if len(hist) > 5 and hist[-6] - obj < 1e-6 * hist[-6]:
                break
            ind = np.zeros(Kd)
            if k > 0:
                ind[np.argsort(-xv, kind="stable")[:k]] = 1.0
            wx.value = 1.0 - ind
            U.value = np.outer(u, u.conj())
            if not solve(p1):
                return False, None, None
        return False, np.maximum(x.value, 0), top(M.value)[2]

    def ps2(sel):
        """DC algorithm for PS2 on the set sel; returns a unit-norm combiner if a rank-one feasible M is found."""
        msk.value = sel.astype(float); U.value = np.zeros((N, N), complex)
        if not solve(p2):
            return None
        hist = []
        for _ in range(dc_iters):
            Mv, w, u = top(M.value)
            gap = (np.trace(Mv).real - w[-1]) / max(np.trace(Mv).real, 1e-12)
            if gap < tol:
                ok = np.all(gamma * np.abs(Hh[sel] @ u.conj()) ** 2 >= 1 - 1e-5)
                return u if ok else None
            hist.append(gap)
            if len(hist) > 5 and hist[-6] - gap < 1e-6 * hist[-6]:
                return None
            U.value = np.outer(u, u.conj())
            if not solve(p2):
                return None
        return None

    # Step 1: smallest k with zero PS1 objective (bisection), keep its x as the priority guideline
    lo, hi, xs = 0, Kd, None
    while lo < hi:
        k = (lo + hi) // 2
        ok, xv, _ = ps1(k)
        if ok:
            hi, xs = k, xv
        else:
            lo = k + 1
    if xs is None:
        ok, xs, _ = ps1(Kd)
        if xs is None:
            xs = np.zeros(Kd)
    order = np.argsort(-xs, kind="stable")
    # Step 2: largest nested set {pi(k), ..., pi(M)} certified feasible (bisection over k)
    lo, hi, best = 0, Kd - 1, None
    while lo <= hi:
        k = (lo + hi) // 2
        sel = np.zeros(Kd, bool); sel[order[k:]] = True
        a = ps2(sel)
        if a is not None:
            best = (sel, a); hi = k - 1
        else:
            lo = k + 1
    if best is None:
        return np.zeros(Kd, bool), Hh[0].conj() / np.linalg.norm(Hh[0])
    sel, a = best
    return sel, a / np.linalg.norm(a)


Y4_DIR = "y4cache"


def y4_tag(seed, thr, eps, block):
    return f"s{seed}_t{thr}_e{eps}_b{block.start}_n{N_RX}_k{K}_j{JAKES_FD}"


def y4_trace(seed, thr, eps=0.0, block=TEST_SLICE):
    """Selections of [4] for every round of one run.  They depend only on the channel estimates, not on the
    noise level or the model, so they are computed once and stored in y4cache/ (re-used at every SNR)."""
    import os, pickle, warnings
    path = os.path.join(Y4_DIR, y4_tag(seed, thr, eps, block) + ".pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    warnings.filterwarnings("ignore")
    Hall = channel_trace("Y4", seed)[block]
    est_rng = np.random.default_rng(3000 + seed)
    out = []
    for t in range(len(Hall)):
        Hh = mmse_estimate(Hall[t], eps, est_rng)
        out.append(y4_select(Hh, 1.0 / thr ** 2))
    os.makedirs(Y4_DIR, exist_ok=True)
    pickle.dump(out, open(path, "wb"))
    return out


# ------------------------------------------------- [14]: statistics-based regularised policy
@lru_cache(maxsize=None)
def y14_design(sigma2, var_h, Kdev, p_avg):
    """Constants of the policy |b(g)| = c g / (g^2 + lam) (closed-form structure of [14]).
    lam minimises the expected aggregation-MSE proxy
        J(lam) = E[(lam/(x+lam))^2]/K + sigma2 E[x/(x+lam)^2] / (2 K^2 p_avg),  x = |c_hat|^2 ~ Exp(var_h),
    and c meets the average power budget E[c^2 x/(x+lam)^2] = p_avg with equality."""
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    pdf = lambda x: np.exp(-x / var_h) / var_h
    E = lambda f: quad(lambda x: f(x) * pdf(x), 0, np.inf, limit=200)[0]
    def J(loglam):
        lam = np.exp(loglam)
        bias = E(lambda x: (lam / (x + lam)) ** 2)
        pw = E(lambda x: x / (x + lam) ** 2)
        return bias / Kdev + sigma2 * pw / (2 * Kdev ** 2 * p_avg)
    res = minimize_scalar(J, bounds=(np.log(1e-6), np.log(1e3)), method="bounded", options=dict(xatol=1e-4))
    lam = float(np.exp(res.x))
    c = float(np.sqrt(p_avg / E(lambda x: x / (x + lam) ** 2)))
    return lam, c


# ------------------------------------------------- [10]: Theorem 1 of Zhang & Tao
def z10_policy(C, beta, N0):
    """Optimal aggregation levels G_k and sqrt(eta) of problem P1 in [10] (Theorem 1),
    for normalised MSE  beta/(beta+1) sum (G_k-1)^2 + 1/(beta+1) (sum G_k - K)^2 + N0/eta,
    with G_k <= C_k / sqrt(eta).  Returns (G, sqrt_eta)."""
    Kd = len(C)
    order = np.argsort(C); Cs = C[order]
    best = None
    cs1, cs2 = np.cumsum(Cs), np.cumsum(Cs ** 2)
    for l in range(1, Kd + 1):
        s1, s2, m = cs1[l - 1], cs2[l - 1], Kd - l
        num = beta / (beta + 1) * s2 + beta * s1 ** 2 / ((beta + 1) * (beta + m)) + N0
        den = beta * (beta + Kd) * s1 / ((beta + m) * (beta + 1))
        sq = num / den; x = 1.0 / sq
        if m > 0:
            G0 = (beta + Kd - x * s1) / (beta + m)
            if G0 < 0 or not (G0 < Cs[l] * x):
                continue
            Gs = np.concatenate([Cs[:l] * x, np.full(m, G0)])
        else:
            Gs = Cs * x
        V = beta / (beta + 1) * np.sum((Gs - 1) ** 2) + (Gs.sum() - Kd) ** 2 / (beta + 1) + N0 * x * x
        if best is None or V < best[0]:
            best = (V, sq, Gs)
    _, sq, Gs = best
    G = np.empty(Kd); G[order] = Gs
    return G, sq


def mmse_estimate(h, eps, rng):
    """Device-side MMSE channel estimate: h = h_hat + e, e ~ CN(0, eps)."""
    if eps <= 0:
        return h.copy()
    # pilot-based estimate  y_p = h + w, w ~ CN(0, eps/(1-eps)) -> MMSE estimate
    nv = eps / (1.0 - eps)
    yp = h + cn(rng, h.shape, nv)
    return yp / (1.0 + nv)


def beamformer(Hh, thr, iters=40, tau=6.0, mu=0.5):
    """Receive combiner a (|a|=1): maximise soft-min_k |a^H h_hat_k|^2."""
    Hh = np.asarray(Hh)                      # (K, N)
    R = Hh.T @ Hh.conj()                     # sum_k h_k h_k^H  (N x N)
    vals, vecs = np.linalg.eigh(R)
    a = vecs[:, -1]
    for _ in range(iters):
        q = np.abs(Hh.conj() @ a) ** 2
        wts = np.exp(-tau * (q - q.min()))
        wts /= wts.sum()
        g = (Hh.T * wts) @ (Hh.conj() @ a)
        a = a + mu * g
        a /= np.linalg.norm(a)
    return a


# ----------------------------------------------------------- one OTA round
def ota_round(G, c, c_hat, method, thr, snr_db, rng, eps=0.0, part_mask=None):
    """
    G : (K, d) true local gradients
    c, c_hat : (K,) true / estimated effective scalar channels
    part_mask : participating set chosen outside (used by Y4)
    Returns g_hat, info
    """
    if NORM and SCALING == "max":   # common scaling by the largest norm: unbiased, symbol power below 1 for most devices
        rho_c = float(np.sqrt(np.sum(G ** 2, 1) / D).max()) + 1e-12
        rho_k = np.full(K, rho_c)
        S = G / rho_c
    elif NORM:   # unit-power symbols: each device scales by its own rho_k, side information reaches the server error-free
        rho_k = np.sqrt(np.sum(G ** 2, 1) / D) + 1e-12
        S = G / rho_k[:, None]
    else:
        rho_k = np.full(K, RHO)
        S = G / RHO
    sigma2 = P_MAX / (10 ** (snr_db / 10))
    gain = np.abs(c_hat)
    pw = np.sum(S ** 2, 1) / D                    # per-element symbol power
    if part_mask is not None:
        part = np.asarray(part_mask, bool)
    elif method in ("M2", "M3", "M2-EIG"):
        part = gain >= thr
    else:                                         # M1, M1-BF, OPT, RCI, C22, Z10, Y14
        part = np.ones(K, bool)
    n_part = int(part.sum())
    if n_part == 0:
        return np.zeros(D), dict(part=0.0, power=0.0, peak=0.0, noise_var=0.0, rx_snr_db=float("nan"),
                                 sqrt_eta=float("nan"))
    b = np.zeros(K, complex)
    scale = None                                  # server multiplier on Re(y) when it is not rho_bar/(|S| sqrt_eta)
    if method == "C22":
        # Case I policy of Cao et al. [12] with the maximum-power constraint (dual variables = 0):
        # every device transmits, received amplitude m_k = min(c, a_k), a_k = |c_hat_k| sqrt(P_MAX/pi_k);
        # c minimises  kappa*(sum_k m_k - K*tau)^2 + sum_k (m_k - tau)^2  (first-order condition solved by bisection).
        # tau is the fixed (noise-agnostic) design target; the server scales by 1/(K*tau).
        tau, kap = thr, C22_KAPPA / K
        a_k = gain * np.sqrt(P_MAX / pw)
        lo, hi = 0.0, tau * (1 + kap * K) + 1e-12
        for _ in range(60):
            c0 = 0.5 * (lo + hi)
            f = c0 - tau + kap * (np.minimum(c0, a_k).sum() - K * tau)
            lo, hi = (c0, hi) if f < 0 else (lo, c0)
        m = np.minimum(0.5 * (lo + hi), a_k)
        b = m * np.conj(c_hat) / gain ** 2
        sqrt_eta = tau
    elif method == "RCI":
        # regularised channel inversion (the form reported in [12]); thr is used as the regulariser lambda.
        lam = thr
        amp = np.sqrt(P_MAX / pw) * (gain ** 2 + lam) / gain
        sqrt_eta = amp.min()
        b = sqrt_eta * np.conj(c_hat) / (gain ** 2 + lam)
    elif method == "OPT":
        # MSE-optimised policy with the threshold structure of Cao et al. [7]:
        # devices that cannot reach the common amplitude sqrt(eta) transmit at full power,
        # the others invert their channel.  eta is chosen by grid search on the
        # expected-error proxy  J(eta) = sum_k pi_k (1 - a_k/sqrt(eta))^2 + sigma^2/(2 eta).
        a_cap = gain * np.sqrt(P_MAX / pw)
        grid = np.geomspace(a_cap.min() * 0.5, a_cap.max() * 2.0, 200)
        J = [np.sum(pw * (1 - np.minimum(a_cap, s) / s) ** 2) + sigma2 / (2 * s * s) for s in grid]
        sqrt_eta = grid[int(np.argmin(J))]
        mag = np.minimum(sqrt_eta / gain, np.sqrt(P_MAX / pw))
        b = mag * np.conj(c_hat) / gain
    elif method == "Z10":
        # Zhang & Tao [10], Theorem 1: devices normalise by the common mean squared gradient norm alpha,
        # the server knows alpha and the squared coefficient of variation beta of the gradients
        # (error-free side information, as for rho_k) and all channel amplitudes.
        alpha = float(np.mean(np.sum(G ** 2, 1)))
        m_d = G.mean(0)
        beta = float(G.var(0).sum() / max(float(m_d @ m_d), 1e-30))
        S = G * np.sqrt(D / alpha)
        pw = np.sum(S ** 2, 1) / D
        Cap = gain * np.sqrt(P_MAX / pw)
        Gk, sqrt_eta = z10_policy(Cap, beta, sigma2 / 2.0)
        b = Gk * sqrt_eta * np.conj(c_hat) / gain ** 2
        scale = np.sqrt(alpha / D) / (K * sqrt_eta)
    elif method == "Y14":
        # closed-form structure of Yu et al. [14]: |b(g)| = c g/(g^2 + lam), constants from the fading statistics,
        # long-term average power budget P_MAX (no per-round peak limit, as in [14]).
        lam, cc = y14_design(round(sigma2, 12), round(1.0 - eps, 12), K, P_MAX)
        b = cc * np.conj(c_hat) / (gain ** 2 + lam)
        if Y14_CAP:                            # per-round peak limit, as for every other policy
            b = b * np.minimum(1.0, np.sqrt(P_MAX / (np.abs(b) ** 2 * pw)))
        sqrt_eta = cc
    else:                                          # M1, M2, M3, M1-BF, M2-EIG, Y4: channel inversion on the set
        sqrt_eta = np.min(np.sqrt(P_MAX / pw[part]) * gain[part])
        b[part] = sqrt_eta * np.conj(c_hat[part]) / gain[part] ** 2
    y = (c[part] * b[part]) @ S[part] + cn(rng, D, sigma2)   # superposition
    rho_bar = float(np.mean(rho_k[part]))
    if DEBIAS:     # unit-gain normalisation: divide by the sum of the estimated received amplitudes
        amp_sum = float(np.sum(np.abs(c_hat[part] * b[part])))
        if PILOT:   # one pilot symbol sent with the precoders through the true channel: |sum_k c_k b_k + n|
            amp_sum = float(np.abs(np.sum(c[part] * b[part]) + cn(PILOT_RNG, 1, sigma2)[0]))
        scale = (np.sqrt(alpha / D) if method == "Z10" else rho_bar) / amp_sum
    if scale is None:
        g_hat = rho_bar * np.real(y) / (n_part * sqrt_eta)
        noise_var = rho_bar ** 2 * D * sigma2 / 2.0 / (n_part * sqrt_eta) ** 2
    else:
        g_hat = scale * np.real(y)
        noise_var = scale ** 2 * D * sigma2 / 2.0
    p_dev = np.abs(b[part]) ** 2 * pw[part]
    tx_power = float(np.mean(p_dev))
    rx = float(np.mean(np.abs(c[part] * b[part]) ** 2 * pw[part]) / sigma2)   # realised receive SNR per device
    return g_hat, dict(part=n_part / K, power=tx_power / P_MAX, peak=float(p_dev.max()) / P_MAX,
                       noise_var=noise_var, rx_snr_db=10 * np.log10(rx), sqrt_eta=float(sqrt_eta))


# ------------------------------------------------------------------- one run
def run(method, snr_db, seed, thr=0.3, eps=0.0, rounds=ROUNDS, block=TEST_SLICE,
        ideal=False):
    Xtr, ytr, Xte, yte = load_data(seed)
    rng = np.random.default_rng(2000 + seed)
    parts = partition(ytr, K, DIRICHLET_ALPHA, rng)
    w = np.zeros(D)
    # channels: P1 trace for N=1 ; same recipe with N_RX columns for the multi-antenna methods
    Hall = channel_trace(method, seed)[block]
    est_rng = np.random.default_rng(3000 + seed)
    noise_rng = np.random.default_rng(4000 + seed)
    global PILOT_RNG
    PILOT_RNG = np.random.default_rng(5000 + seed)
    y4 = y4_trace(seed, thr, eps, block) if (method == "Y4" and not ideal) else None
    log = dict(acc=[], mse=[], part=[], power=[], peak=[], noise_var=[], rx_snr_db=[], sqrt_eta=[], proj=[], cos=[])
    for t in range(rounds):
        G = np.stack([grad(w, Xtr[p], ytr[p]) for p in parts])
        g_bar = G.mean(0)
        pm = None
        if ideal:
            g_hat, info = g_bar, dict(part=1.0, power=0.0, peak=0.0, noise_var=0.0, rx_snr_db=float("nan"),
                                      sqrt_eta=float("nan"))
        else:
            H = Hall[t]
            if H.ndim == 1:
                c, c_hat = H, mmse_estimate(H, eps, est_rng)
            else:
                Hh = mmse_estimate(H, eps, est_rng)
                if method == "Y4":           # thr is the threshold h_th on |a^H h_hat|, i.e. gamma = 1/h_th^2
                    pm, a = y4[t]
                else:
                    a = beamformer(Hh, thr, iters=(0 if method == "M2-EIG" else 40))
                c, c_hat = H @ a.conj(), Hh @ a.conj()
            g_hat, info = ota_round(G, c, c_hat, method, thr, snr_db, noise_rng, eps=eps, part_mask=pm)
        err = g_hat - g_bar
        log["mse"].append(float(err @ err))
        gg = max(float(g_bar @ g_bar), 1e-30)
        log["proj"].append(float(g_hat @ g_bar) / gg)   # shrinkage of the estimate along g_bar
        log["cos"].append(float(g_hat @ g_bar) / np.sqrt(gg * max(float(g_hat @ g_hat), 1e-30)))
        for k in ("part", "power", "peak", "noise_var", "rx_snr_db", "sqrt_eta"):
            log[k].append(info.get(k, float("nan")))
        w = w - LR * g_hat
        log["acc"].append(accuracy(w, Xte, yte))
    return log


def avg_over_seeds(method, snr_db, thr=0.3, eps=0.0, seeds=SEEDS, block=TEST_SLICE,
                   ideal=False):
    logs = [run(method, snr_db, s, thr, eps, block=block, ideal=ideal) for s in seeds]
    out = {}
    for k in logs[0]:
        out[k] = np.nanmean([l[k] for l in logs], 0) if k == "rx_snr_db" else np.mean([l[k] for l in logs], 0)
    out["acc_final_all"] = [l["acc"][-1] for l in logs]
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = {}
    SNRS = [-35, -30, -25, -20, -15, -10, 0, 10, 20, 30]

    # ---- tuning on P1 *train* rounds (0-99): threshold for M2 and M3 -------
    THRS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.2]
    tune = {"M2": {}, "M3": {}}
    for m in ("M2", "M3"):
        for th in THRS:
            r = avg_over_seeds(m, SNR_REF, thr=th, seeds=SEEDS[:5], block=TRAIN_SLICE)
            tune[m][th] = float(np.mean(r["acc"][-10:]))
    best_thr = {m: max(tune[m], key=tune[m].get) for m in tune}
    R["tuning"] = {m: {str(k): v for k, v in tune[m].items()} for m in tune}
    R["best_thr"] = best_thr
    print("tuned thresholds (train rounds):", best_thr)

    TH = best_thr
    methods = [("M1", dict(thr=0.0)), ("M2", dict(thr=TH["M2"])),
               ("M3", dict(thr=TH["M3"])), ("M1-BF", dict(thr=0.0))]

    # ---- A: learning curves @ SNR_REF, perfect CSI ---------------------------
    R["curve"] = {}
    ideal = avg_over_seeds("M1", SNR_REF, ideal=True)
    R["curve"]["IDEAL"] = ideal["acc"].tolist()
    for m, kw in methods:
        r = avg_over_seeds(m, SNR_REF, **kw)
        R["curve"][m] = r["acc"].tolist()
        R["curve"][m + "_mse"] = r["mse"].tolist()

    # ---- B: sweep over SNR (perfect CSI) -----------------------------------
    R["snr"] = {"snrs": SNRS}
    R["snr"]["IDEAL_acc"] = float(np.mean(ideal["acc"][-10:]))
    for m, kw in methods:
        R["snr"][m] = dict(acc=[], acc_std=[], mse=[], part=[], power=[], noise_var=[])
        for s in SNRS:
            r = avg_over_seeds(m, s, **kw)
            R["snr"][m]["acc"].append(float(np.mean(r["acc"][-10:])))
            R["snr"][m]["acc_std"].append(float(np.std(r["acc_final_all"])))
            R["snr"][m]["mse"].append(float(np.mean(r["mse"])))
            R["snr"][m]["part"].append(float(np.mean(r["part"])))
            R["snr"][m]["power"].append(float(np.mean(r["power"])))
            R["snr"][m]["noise_var"].append(float(np.mean(r["noise_var"])))
        print(m, "acc vs SNR", np.round(R["snr"][m]["acc"], 3))

    # ---- C: imperfect CSI sweep @ SNR_REF ------------------------------------
    EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
    R["csi"] = {"eps": EPS}
    for m, kw in methods:
        R["csi"][m] = dict(acc=[], mse=[])
        for e in EPS:
            r = avg_over_seeds(m, SNR_REF, eps=e, **kw)
            R["csi"][m]["acc"].append(float(np.mean(r["acc"][-10:])))
            R["csi"][m]["mse"].append(float(np.mean(r["mse"])))
        print(m, "acc vs eps", np.round(R["csi"][m]["acc"], 3))

    # ---- D: M2 threshold sweep on test block: bias-variance ----------------
    R["thr"] = {"thr": THRS, "acc": [], "mse": [], "part": [], "noise_var": []}
    for th in THRS:
        r = avg_over_seeds("M2", SNR_REF, thr=th)
        R["thr"]["acc"].append(float(np.mean(r["acc"][-10:])))
        R["thr"]["mse"].append(float(np.mean(r["mse"])))
        R["thr"]["part"].append(float(np.mean(r["part"])))
        R["thr"]["noise_var"].append(float(np.mean(r["noise_var"])))

    with open("results.json", "w") as f:
        json.dump(R, f, indent=1)

    # ----------------------------------------------------------- figures
    sty = {"M1": ("k", "-", "o"), "M2": ("tab:blue", "--", "s"),
           "M3": ("tab:red", "-", "^"), "M1-BF": ("tab:green", ":", "d"),
           "IDEAL": ("gray", "-.", None)}
    lab = {"M1": "M1 (full inversion)", "M2": "M2 (truncated inversion)",
           "M3": "M3 (beamforming + truncation)", "M1-BF": "M1 + beamforming only",
           "IDEAL": "Ideal (noiseless)"}
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in ["IDEAL", "M1", "M2", "M3"]:
        c, ls, _ = sty[m]
        ax.plot(range(1, ROUNDS + 1), R["curve"][m], color=c, ls=ls, label=lab[m])
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy")
    ax.legend(fontsize=7.5); fig.tight_layout(); fig.savefig("fig_curves.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in ["M1", "M2", "M3", "M1-BF"]:
        c, ls, mk = sty[m]
        ax.errorbar(SNRS, R["snr"][m]["acc"], yerr=R["snr"][m]["acc_std"], color=c,
                    ls=ls, marker=mk, ms=4, capsize=2, label=lab[m])
    ax.axhline(R["snr"]["IDEAL_acc"], color="gray", ls="-.", label=lab["IDEAL"])
    ax.set_xlabel("SNR  P/$\\sigma^2$ (dB)"); ax.set_ylabel("Test accuracy")
    ax.legend(fontsize=7.5); fig.tight_layout(); fig.savefig("fig_snr.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in ["M1", "M2", "M3", "M1-BF"]:
        c, ls, mk = sty[m]
        ax.semilogy(SNRS, R["snr"][m]["mse"], color=c, ls=ls, marker=mk, ms=4,
                    label=lab[m])
    ax.set_xlabel("SNR  P/$\\sigma^2$ (dB)"); ax.set_ylabel("Aggregation MSE")
    ax.legend(fontsize=7.5); fig.tight_layout(); fig.savefig("fig_mse.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in ["M1", "M2", "M3", "M1-BF"]:
        c, ls, mk = sty[m]
        ax.plot(EPS, R["csi"][m]["acc"], color=c, ls=ls, marker=mk, ms=4, label=lab[m])
    ax.set_xlabel("CSI error variance $\\epsilon$"); ax.set_ylabel("Test accuracy")
    ax.legend(fontsize=7.5); fig.tight_layout(); fig.savefig("fig_csi.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.semilogy(THRS, R["thr"]["mse"], "o-", color="tab:blue", label="Aggregation MSE")
    ax.semilogy(THRS, np.maximum(R["thr"]["noise_var"], 1e-12), "s--", color="tab:orange",
                label="Noise-only variance")
    ax.set_xlabel("Truncation threshold $h_{th}$"); ax.set_ylabel("Error")
    ax2 = ax.twinx(); ax2.plot(THRS, R["thr"]["part"], "^:", color="tab:green",
                               label="Participation ratio"); ax2.set_ylabel("Participation ratio")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.5); fig.tight_layout()
    fig.savefig("fig_threshold.png", dpi=200)
    print("done")
