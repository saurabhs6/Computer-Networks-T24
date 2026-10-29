"""Figures of the revised manuscript. Uses the same data as the text (num_v3)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import num_v3 as NV

OUT = os.path.join(NV.SIM, "figs"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 6.8,
                     "axes.titlesize": 8.5})
STY = {"M1": ("k", "-", "o"), "M2": ("tab:blue", "--", "s"), "M3": ("tab:red", "-", "^"), "M2-EIG": ("tab:purple", "-.", "v"),
       "M1-BF": ("tab:green", ":", "d"), "OPT": ("tab:orange", ":", "x"), "RCI": ("tab:brown", "-.", "+"),
       "C22": ("tab:cyan", "-", "P"), "Z10": ("tab:olive", "--", "*"), "Y14": ("tab:pink", ":", "h"), "Y4": ("dimgray", "--", "D"),
       "OPT-U": ("tab:orange", "-", ">"), "RCI-U": ("tab:brown", "-", "<"), "Z10-U": ("tab:olive", "-", "p"),
       "Y14-U": ("tab:pink", "-", "H"), "IDEAL": ("gray", "-.", None)}
M = NV.main
SN = NV.REF
SNRS = sorted(s for (m, s) in M if m == "M2" and isinstance(s, (int, float)))
IDEAL = NV.IDEAL
nm = NV.nm


def line(ax, m, snrs=None, g=M, err=True):
    snrs = snrs or sorted(s for (k, s) in g if k == m and isinstance(s, (int, float)))
    mu, sd = zip(*[NV.A(g, (m, s)) for s in snrs])
    c, ls, mk = STY[m]
    open_ = m.endswith("-U")
    kw = dict(color=c, ls=ls, marker=mk, ms=3.3, lw=1.1, label=nm(m), mfc="none" if open_ else c)
    if err:
        ax.errorbar(snrs, mu, yerr=sd, capsize=1.5, **kw)
    else:
        ax.plot(snrs, mu, **kw)


# Fig 2: accuracy vs SNR, three panels
fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.3))
for m in ["M1", "M1-BF", "M2", "M2-EIG", "M3"]:
    line(axs[0], m)
for m in ["M2", "OPT", "RCI", "C22", "Z10", "Y14", "Y4"]:
    line(axs[1], m)
for m in ["C22", "OPT-U", "RCI-U", "Z10-U", "Y14-U"]:
    line(axs[2], m)
for ax, t in zip(axs, ["(a) Truncation and receive combining", "(b) Policies from the literature",
                       "(c) Unit-gain normalisation"]):
    ax.axhline(IDEAL, color="gray", ls="-.", lw=1, label="Ideal")
    ax.set_xlabel("Per-device transmit SNR (dB)"); ax.set_title(t)
    ax.legend(loc="lower right", ncol=2)
axs[0].set_ylim(10, 97); axs[1].set_ylim(50, 96); axs[2].set_ylim(50, 96)
for ax in axs:
    ax.set_ylabel("Test accuracy (%)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_snr.png", dpi=200); plt.close(fig)

# Fig 3: learning curves at the reference SNR
fig, ax = plt.subplots(figsize=(5.4, 3.4))
for m in ["IDEAL", "M1", "M2", "M3", "C22", "OPT", "OPT-U", "Z10", "Y14", "Y14-U"]:
    key = (m, SN) if "curve" in next(iter(M[(m, SN)].values())) else (m, "curve")
    cur = np.mean([r["curve"] for r in M[key].values()], 0) * 100
    c, ls, _ = STY[m]
    ax.plot(range(1, len(cur) + 1), cur, color=c, ls=ls, lw=1.2, label="Ideal" if m == "IDEAL" else nm(m))
ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 97)
ax.legend(ncol=2, loc="center right", bbox_to_anchor=(1.0, 0.42)); fig.tight_layout(); fig.savefig(f"{OUT}/fig_curves.png", dpi=200); plt.close(fig)

# Fig 4: aggregation MSE vs SNR, mean and median
fig, ax = plt.subplots(figsize=(5.6, 3.6))
for m in ["M1", "M2", "M3", "C22", "OPT", "Z10"]:
    c, ls, mk = STY[m]
    mu = [NV.F(M, (m, s), "mse_mean") for s in SNRS]
    md = [NV.F(M, (m, s), "mse_med") for s in SNRS]
    ax.semilogy(SNRS, mu, color=c, ls="-", marker=mk, ms=3.3, lw=1.1, label=f"{nm(m)} mean")
    ax.semilogy(SNRS, md, color=c, ls=":", marker=mk, ms=3.3, mfc="none", lw=1.0, label=f"{nm(m)} median")
ax.set_xlabel("Per-device transmit SNR (dB)"); ax.set_ylabel("Aggregation MSE per round")
ax.legend(ncol=3, fontsize=6.0); fig.tight_layout(); fig.savefig(f"{OUT}/fig_mse.png", dpi=200); plt.close(fig)

# Fig 5: accuracy against MSE and against shrinkage at the reference SNR
fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True)
for m in ["M1-BF", "M2", "M2-EIG", "M3", "Y4", "OPT", "RCI", "C22", "Z10", "Y14", "OPT-U", "RCI-U", "Z10-U", "Y14-U"]:
    if (m, SN) not in M:
        continue
    c, _, mk = STY[m]
    a = NV.av(m); e = NV.F(M, (m, SN), "mse_mean"); p = NV.F(M, (m, SN), "proj")
    kw = dict(color=c, marker=mk, s=34, linewidths=1.2)
    if m.endswith("-U") and mk not in ("x", "+"):
        kw.update(facecolors="none", edgecolors=c); kw.pop("color")
    axs[0].scatter(e, a, label=nm(m), **kw)
    axs[1].scatter(p, a, **kw)
axs[0].set_xscale("log"); axs[0].set_xlabel("Mean aggregation MSE"); axs[0].set_ylabel("Test accuracy (%)")
axs[1].set_xlabel("Shrinkage factor $\\langle\\hat g,\\bar g\\rangle/\\|\\bar g\\|^2$")
axs[0].set_title("(a) Accuracy against MSE"); axs[1].set_title("(b) Accuracy against shrinkage")
fig.legend(loc="center right", ncol=1, fontsize=6.6, frameon=True)
fig.tight_layout(rect=(0, 0, 0.86, 1)); fig.savefig(f"{OUT}/fig_shrink.png", dpi=200); plt.close(fig)

# Fig 6: tail of the per-round MSE at the reference SNR
fig, ax = plt.subplots(figsize=(5.4, 3.3))
for m in ["M1", "M1-BF", "M2", "M3", "C22"]:
    v = np.sort(np.concatenate([r["mse_curve"] for r in M[(m, SN)].values()]))
    c, ls, _ = STY[m]
    ax.loglog(v, 1 - np.arange(len(v)) / len(v), color=c, ls=ls, lw=1.3, label=nm(m))
ax.set_ylim(1e-3, 1.05)
ax.set_xlabel("Aggregation MSE in one round"); ax.set_ylabel("Fraction of rounds above")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig_tail.png", dpi=200); plt.close(fig)

# Fig 7: accuracy against realised receive SNR
fig, ax = plt.subplots(figsize=(5.4, 3.5))
for m in ["M1", "M1-BF", "M2", "M3", "OPT", "RCI", "C22", "Z10"]:
    c, ls, mk = STY[m]
    xs = [NV.F(M, (m, s), "rx") for s in SNRS]
    ys = [NV.av(m, s) for s in SNRS]
    ax.plot(xs, ys, color=c, ls=ls, marker=mk, ms=3.3, lw=1.1, label=nm(m))
ax.axhline(IDEAL, color="gray", ls="-.", lw=1, label="Ideal")
ax.set_xlim(-65, 25); ax.set_ylim(10, 97)
ax.set_xlabel("Realised receive SNR (dB)"); ax.set_ylabel("Test accuracy (%)"); ax.legend(ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_rxsnr.png", dpi=200); plt.close(fig)

# Fig 8: CSI sweep
EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
fig, ax = plt.subplots(figsize=(5.4, 3.5))
for m in ["M1-BF", "M2", "M3", "OPT", "RCI", "C22", "Z10", "Y14", "OPT-U", "Y14-U", "Y4"]:
    es = [e for e in EPS if (m, e) in NV.csi]
    c, ls, mk = STY[m]
    ax.plot(es, [NV.A(NV.csi, (m, e))[0] for e in es], color=c, ls=ls, marker=mk, ms=3.3, lw=1.1, label=nm(m),
            mfc="none" if m.endswith("-U") else c)
ax.set_xlabel("CSI error variance $\\epsilon$"); ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(60, 95)
ax.legend(ncol=3, loc="lower left"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_csi.png", dpi=200); plt.close(fig)

# Fig 9: threshold sweep of M2
TH = sorted(t for (t,) in NV.thr)
fig, ax = plt.subplots(figsize=(5.4, 3.5))
ax.semilogy(TH, [NV.F(NV.thr, (t,), "mse_mean") for t in TH], "o-", color="tab:blue", label="Aggregation MSE")
ax.semilogy(TH, [NV.F(NV.thr, (t,), "noise") for t in TH], "s--", color="tab:orange", label="Noise-only variance")
ax.set_xlabel("Truncation threshold $h_{th}$"); ax.set_ylabel("Error")
ax2 = ax.twinx(); ax2.plot(TH, [NV.F(NV.thr, (t,), "part") for t in TH], "^:", color="tab:green", label="Participation ratio")
ax2.set_ylabel("Participation ratio"); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2); fig.tight_layout(); fig.savefig(f"{OUT}/fig_threshold.png", dpi=200); plt.close(fig)

# Fig 10: CIFAR-10 learning curves at the reference SNR
if NV.CF:
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for m in ["IDEAL", "M1", "M2", "M3", "C22", "OPT", "OPT-U", "Z10", "Z10-U", "Y14", "Y14-U"]:
        key = (m, float(SN))
        if key not in NV.CF or not NV.CF[key]:
            continue
        cur = np.mean([r["acc"] for r in NV.CF[key].values()], 0) * 100
        x = np.arange(1, len(cur) + 1) * 25
        c, ls, mk = STY[m]
        ax.plot(x, cur, color=c, ls=ls, marker=mk, ms=3, lw=1.1, label="Ideal" if m == "IDEAL" else nm(m),
                mfc="none" if m.endswith("-U") else c)
    ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 60)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5)); fig.tight_layout(); fig.savefig(f"{OUT}/fig_cifar.png", dpi=200); plt.close(fig)
print("figures written to", OUT)
