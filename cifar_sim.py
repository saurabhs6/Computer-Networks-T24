"""Realistic-task check of the revised study: CIFAR-10, a CNN with 136,874 parameters (PyTorch),
K = 50 devices with Dirichlet(0.5) label skew, unit-power symbols, the same over-the-air aggregation
(ota_fl_sim.ota_round) as the digits and MNIST tasks."""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, grad, vmap
import ota_fl_sim as S

K_DEV, ROUNDS, BATCH, EVAL_EVERY = 50, 300, 16, 25
DATA = os.environ.get("CIFAR_NPZ", "../data/cifar10.npz")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.f1 = nn.Linear(32 * 8 * 8, 64)
        self.f2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        return self.f2(F.relu(self.f1(x.flatten(1))))


def load():
    z = np.load(DATA)
    xtr = z["train_x"].astype(np.float32) / 255.0; xte = z["test_x"].astype(np.float32) / 255.0
    mu = xtr.mean((0, 1, 2)); sd = xtr.std((0, 1, 2))
    f = lambda x: torch.from_numpy(((x - mu) / sd).transpose(0, 3, 1, 2).copy())
    return f(xtr), torch.from_numpy(z["train_y"].astype(np.int64)), f(xte), torch.from_numpy(z["test_y"].astype(np.int64))


def channels(method, seed, n):
    multi = method in ("M3", "M1-BF", "M2-EIG")
    rng = np.random.default_rng(7000 + seed + (100 if multi else 0))
    return S.cn(rng, (n, K_DEV, S.N_RX) if multi else (n, K_DEV), 1.0)


def run(method, snr_db, seed, thr, data, lr, eps=0.0, ideal=False, rounds=ROUNDS, kappa=100.0):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = data
    net = Net()
    names = [n for n, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    sizes = [p.numel() for _, p in net.named_parameters()]
    d = sum(sizes)
    params = {n: p.detach().clone() for n, p in net.named_parameters()}
    S.K, S.D, S.NORM, S.C22_KAPPA = K_DEV, d, True, kappa
    parts = S.partition(ytr.numpy(), K_DEV, 0.5, np.random.default_rng(2000 + seed))
    brng = np.random.default_rng(5000 + seed)
    Hall = channels(method, seed, rounds)
    est_rng = np.random.default_rng(3000 + seed); noise_rng = np.random.default_rng(4000 + seed)

    def loss(p, x, y):
        return F.cross_entropy(functional_call(net, p, (x,)), y)
    pgrad = vmap(grad(loss), in_dims=(None, 0, 0))

    def evaluate():
        with torch.no_grad():
            out = [functional_call(net, params, (Xte[i:i + 2500],)).argmax(1) for i in range(0, len(Xte), 2500)]
        return float((torch.cat(out) == yte).float().mean())

    acc, mse, part, rx, proj = [], [], [], [], []
    for t in range(rounds):
        idx = np.stack([brng.choice(p, BATCH, replace=len(p) < BATCH) for p in parts])
        gd = pgrad(params, Xtr[idx], ytr[idx])
        G = torch.cat([gd[n].reshape(K_DEV, -1) for n in names], 1).double().numpy()
        g_bar = G.mean(0)
        if ideal:
            g_hat, info = g_bar, dict(part=1.0, rx_snr_db=np.nan)
        else:
            H = Hall[t]
            if H.ndim == 1:
                c, c_hat = H, S.mmse_estimate(H, eps, est_rng)
            else:
                Hh = S.mmse_estimate(H, eps, est_rng)
                a = S.beamformer(Hh, thr, iters=(0 if method == "M2-EIG" else 40))
                c, c_hat = H @ a.conj(), Hh @ a.conj()
            g_hat, info = S.ota_round(G, c, c_hat, method, thr, snr_db, noise_rng, eps=eps)
        err = g_hat - g_bar
        mse.append(float(err @ err)); part.append(info["part"]); rx.append(info.get("rx_snr_db", np.nan))
        proj.append(float(g_hat @ g_bar) / float(g_bar @ g_bar))
        upd = torch.from_numpy(g_hat).float()
        o = 0
        for n, sh, sz in zip(names, shapes, sizes):
            params[n] = params[n] - lr * upd[o:o + sz].reshape(sh); o += sz
        if (t + 1) % EVAL_EVERY == 0:
            acc.append(evaluate())
    m = np.array(mse)
    return dict(acc=acc, mse_mean=float(m.mean()), mse_med=float(np.median(m)), mse_p90=float(np.percentile(m, 90)),
                part=float(np.mean(part)), rx=float(np.nanmean(rx)) if not ideal else None, proj=float(np.mean(proj)), d=d)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    data = load()
    t = time.time()
    r = run(sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), data, float(sys.argv[5]),
            ideal=(sys.argv[1] == "IDEAL"), rounds=int(sys.argv[6]) if len(sys.argv) > 6 else ROUNDS)
    print(json.dumps(r), round(time.time() - t, 1), "s")
