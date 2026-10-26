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
"""
import json
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
def ota_round(G, c, c_hat, method, thr, snr_db, rng):
    """
    G : (K, d) true local gradients
    c, c_hat : (K,) true / estimated effective scalar channels
    Returns g_hat, info
    """
    if NORM:   # unit-power symbols: each device scales by its own rho_k, side information reaches the server error-free
        rho_k = np.sqrt(np.sum(G ** 2, 1) / D) + 1e-12
        S = G / rho_k[:, None]
    else:
        rho_k = np.full(K, RHO)
        S = G / RHO
    sigma2 = P_MAX / (10 ** (snr_db / 10))
    gain = np.abs(c_hat)
    pw = np.sum(S ** 2, 1) / D                    # per-element symbol power
    if method in ("M2", "M3", "M2-EIG"):
        part = gain >= thr
    else:                                         # M1, M1-BF, OPT
        part = np.ones(K, bool)
    n_part = int(part.sum())
    if n_part == 0:
        return np.zeros(D), dict(part=0.0, power=0.0, noise_var=0.0, rx_snr_db=float("nan"))
    b = np.zeros(K, complex)
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
    else:
        sqrt_eta = np.min(np.sqrt(P_MAX / pw[part]) * gain[part])
        b[part] = sqrt_eta * np.conj(c_hat[part]) / gain[part] ** 2
    y = (c[part] * b[part]) @ S[part] + cn(rng, D, sigma2)   # superposition
    rho_bar = float(np.mean(rho_k[part]))
    g_hat = rho_bar * np.real(y) / (n_part * sqrt_eta)
    tx_power = float(np.mean(np.abs(b[part]) ** 2 * pw[part]))
    noise_var = rho_bar ** 2 * D * sigma2 / 2.0 / (n_part * sqrt_eta) ** 2
    rx = float(np.mean(np.abs(c[part] * b[part]) ** 2 * pw[part]) / sigma2)   # realised receive SNR per device
    return g_hat, dict(part=n_part / K, power=tx_power / P_MAX, noise_var=noise_var,
                       rx_snr_db=10 * np.log10(rx))


# ------------------------------------------------------------------- one run
def run(method, snr_db, seed, thr=0.3, eps=0.0, rounds=ROUNDS, block=TEST_SLICE,
        ideal=False):
    Xtr, ytr, Xte, yte = load_data(seed)
    rng = np.random.default_rng(2000 + seed)
    parts = partition(ytr, K, DIRICHLET_ALPHA, rng)
    w = np.zeros(D)
    # channels: P1 trace for N=1 ; same recipe with N_RX columns for M3
    if method in ("M3", "M1-BF", "M2-EIG"):
        crng = np.random.default_rng(P1_SEED + seed + 77)
        Hall = cn(crng, (500, K, N_RX), 1.0)[block]
    else:
        Hall = p1_channels(seed)[block]
    est_rng = np.random.default_rng(3000 + seed)
    noise_rng = np.random.default_rng(4000 + seed)
    log = dict(acc=[], mse=[], part=[], power=[], noise_var=[], bias2=[], rx_snr_db=[])
    for t in range(rounds):
        G = np.stack([grad(w, Xtr[p], ytr[p]) for p in parts])
        g_bar = G.mean(0)
        if ideal:
            g_hat, info = g_bar, dict(part=1.0, power=0.0, noise_var=0.0, rx_snr_db=float("nan"))
        else:
            H = Hall[t]
            if H.ndim == 1:
                c, c_hat = H, mmse_estimate(H, eps, est_rng)
            else:
                Hh = mmse_estimate(H, eps, est_rng)
                a = beamformer(Hh, thr, iters=(0 if method == "M2-EIG" else 40))
                c, c_hat = H @ a.conj(), Hh @ a.conj()
            g_hat, info = ota_round(G, c, c_hat, method, thr, snr_db, noise_rng)
        err = g_hat - g_bar
        log["mse"].append(float(err @ err))
        log["part"].append(info["part"])
        log["power"].append(info["power"])
        log["noise_var"].append(info["noise_var"])
        log["rx_snr_db"].append(info.get("rx_snr_db", float("nan")))
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
