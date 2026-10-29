"""Larger-task check: MNIST, small CNN (13,610 parameters, written in NumPy), K = 100 devices.
Re-uses the over-the-air aggregation of ota_fl_sim.py (M1, M2, M3, M2+EIG, OPT, RCI)."""
import gzip, json, sys, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
import ota_fl_sim as S

K_DEV, ROUNDS, BATCH, LR_C = 100, 100, 32, 0.5
DC = 8                                   # conv filters
D_C = 9 * DC + DC + 13 * 13 * DC * 10 + 10   # 13,610


def load():
    def img(f):
        with gzip.open(f) as fh: return np.frombuffer(fh.read(), np.uint8, offset=16).reshape(-1, 28, 28) / 255.0
    def lab(f):
        with gzip.open(f) as fh: return np.frombuffer(fh.read(), np.uint8, offset=8).astype(int)
    return img("train-images-idx3-ubyte.gz"), lab("train-labels-idx1-ubyte.gz"), img("t10k-images-idx3-ubyte.gz"), lab("t10k-labels-idx1-ubyte.gz")


def unpack(w):
    Wc = w[:9 * DC].reshape(9, DC); bc = w[9 * DC:10 * DC]
    o = 10 * DC; Wf = w[o:o + 1352 * 10].reshape(1352, 10); bf = w[o + 13520:]
    return Wc, bc, Wf, bf


def forward(w, X):
    Wc, bc, Wf, bf = unpack(w)
    B = len(X)
    P = swv(X, (3, 3), axis=(1, 2)).reshape(B * 676, 9)
    Z = (P @ Wc + bc).reshape(B, 26, 26, DC)
    A = np.maximum(Z, 0)
    A6 = A.reshape(B, 13, 2, 13, 2, DC)
    H4 = A6.max(axis=(2, 4))
    H = H4.reshape(B, 1352)
    return P, Z, A6, H4, H, H @ Wf + bf


def grad(w, X, y):
    Wc, bc, Wf, bf = unpack(w); B = len(X)
    P, Z, A6, H4, H, L = forward(w, X)
    Pr = S.softmax(L); Pr[np.arange(B), y] -= 1; Pr /= B
    gWf = H.T @ Pr; gbf = Pr.sum(0)
    dH = (Pr @ Wf.T).reshape(B, 13, 13, DC)
    mask = (A6 == H4[:, :, None, :, None, :])
    dA = (mask * dH[:, :, None, :, None, :]).reshape(B, 26, 26, DC)
    dZ = dA * (Z > 0)
    dZf = dZ.reshape(B * 676, DC)
    return np.concatenate([(P.T @ dZf).ravel(), dZf.sum(0), gWf.ravel(), gbf])


def accuracy(w, X, y):
    out = []
    for i in range(0, len(X), 2000):
        out.append(np.argmax(forward(w, X[i:i + 2000])[-1], 1))
    return float(np.mean(np.concatenate(out) == y))


def init(rng):
    w = np.zeros(D_C)
    w[:9 * DC] = rng.normal(0, np.sqrt(2 / 9), 9 * DC)
    o = 10 * DC
    w[o:o + 13520] = rng.normal(0, np.sqrt(1 / 1352), 13520)
    return w


def run(method, snr_db, seed, thr, data, rho, eps=0.0, ideal=False, eval_every=10):
    Xtr, ytr, Xte, yte = data
    S.K, S.D, S.RHO = K_DEV, D_C, rho
    rng = np.random.default_rng(2000 + seed)
    parts = S.partition(ytr, K_DEV, 0.5, rng)
    w = init(np.random.default_rng(10 + seed))
    block = S.TEST_SLICE
    if method in ("M3", "M1-BF", "M2-EIG"):
        Hall = S.cn(np.random.default_rng(S.P1_SEED + seed + 77), (500, K_DEV, S.N_RX), 1.0)[block]
    else:
        Hall = S.p1_channels(seed)[block]
    est_rng = np.random.default_rng(3000 + seed); noise_rng = np.random.default_rng(4000 + seed); brng = np.random.default_rng(5000 + seed)
    acc, mse, part, rx = [], [], [], []
    for t in range(ROUNDS):
        G = np.stack([grad(w, Xtr[idx], ytr[idx]) for idx in (brng.choice(p, min(BATCH, len(p)), replace=False) for p in parts)])
        g_bar = G.mean(0)
        if ideal:
            g_hat = g_bar; info = dict(part=1.0, rx_snr_db=np.nan)
        else:
            H = Hall[t]
            if H.ndim == 1:
                c, c_hat = H, S.mmse_estimate(H, eps, est_rng)
            else:
                Hh = S.mmse_estimate(H, eps, est_rng)
                a = S.beamformer(Hh, thr, iters=(0 if method == "M2-EIG" else 40))
                c, c_hat = H @ a.conj(), Hh @ a.conj()
            g_hat, info = S.ota_round(G, c, c_hat, method, thr, snr_db, noise_rng)
        err = g_hat - g_bar; mse.append(float(err @ err)); part.append(info["part"]); rx.append(info.get("rx_snr_db", np.nan))
        w = w - LR_C * g_hat
        if (t + 1) % eval_every == 0: acc.append(accuracy(w, Xte, yte))
    return dict(acc=acc, mse=float(np.mean(mse)), part=float(np.mean(part)), rx=float(np.nanmean(rx)) if not ideal else float("nan"))


def calibrate_rho(data, target=0.75):
    """rho such that the mean per-element symbol power of the initial local gradients is about `target`."""
    Xtr, ytr, _, _ = data
    rng = np.random.default_rng(1); w = init(np.random.default_rng(10))
    parts = S.partition(ytr, K_DEV, 0.5, np.random.default_rng(2000))
    G = np.stack([grad(w, Xtr[rng.choice(p, 32, replace=False)], ytr[rng.choice(p, 32, replace=False)] if False else ytr[rng.choice(p, 32, replace=False)]) for p in parts[:5]])
    return float(np.sqrt(np.mean(G ** 2) / target))


if __name__ == "__main__":
    data = load()
    t = time.time()
    rho = 0.0
    # simple calibration with matched batches
    Xtr, ytr = data[0], data[1]; rng = np.random.default_rng(1); w0 = init(np.random.default_rng(10))
    parts = S.partition(ytr, K_DEV, 0.5, np.random.default_rng(2000))
    gs = []
    for p in parts[:10]:
        ii = rng.choice(p, 32, replace=False); gs.append(grad(w0, Xtr[ii], ytr[ii]))
    rho = float(np.sqrt(np.mean(np.stack(gs) ** 2) / 0.75))
    print("rho", rho, "d", D_C)
    r = run("M1", -25, 0, 0.0, data, rho, ideal=True)
    print("ideal", r["acc"], round(time.time() - t, 1), "s")
