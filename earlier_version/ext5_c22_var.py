"""Variants of the [12] policy on digits: kappa0 = 0 (closed form of Prop. 3 with zero dual variables)
and kappa0 = 1e4 (unbiased case, kappa -> infinity)."""
import json, numpy as np, ota_fl_sim as S
SNRS = [-35, -25, -15]
E = {}
for kap in (0.0, 1e4):
    S.C22_KAPPA = kap
    tune = {}
    for tau in (0.8, 1.2, 2.0, 3.0, 5.0):
        r = S.avg_over_seeds("C22", S.SNR_REF, thr=tau, seeds=S.SEEDS[:5], block=S.TRAIN_SLICE)
        tune[tau] = float(np.mean(r["acc"][-10:]))
    tau = max(tune, key=tune.get)
    E[str(kap)] = {"tau": tau, "tune": {str(k): v for k, v in tune.items()}, "snr": SNRS, "acc": [], "mse": [], "power": []}
    for s in SNRS:
        r = S.avg_over_seeds("C22", s, thr=tau)
        E[str(kap)]["acc"].append(float(np.mean(r["acc"][-10:]))); E[str(kap)]["mse"].append(float(np.mean(r["mse"])))
        E[str(kap)]["power"].append(float(np.mean(r["power"])))
    print(kap, tau, np.round(E[str(kap)]["acc"], 3), np.round(E[str(kap)]["mse"], 3), flush=True)
    json.dump(E, open("results_ext5.json", "w"), indent=1)
