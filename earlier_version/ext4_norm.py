"""Unit-power-normalised symbols (review item 3): redo tuning and the SNR sweep."""
import json, numpy as np, ota_fl_sim as S
S.NORM = True; S.C22_KAPPA = 10.0
R = json.load(open("results.json")); SNRS = R["snr"]["snrs"]
grids = {"M2": [0.3, 0.5, 0.7, 1.0], "M3": [0.5, 0.7, 1.0], "C22": [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]}
best = {"M1": 0.0, "OPT": 0.0}; tune = {}
for m, g in grids.items():
    tune[m] = {}
    for th in g:
        r = S.avg_over_seeds(m, S.SNR_REF, thr=th, seeds=S.SEEDS[:5], block=S.TRAIN_SLICE)
        tune[m][th] = float(np.mean(r["acc"][-10:]))
    best[m] = max(tune[m], key=tune[m].get); print(m, tune[m], "best", best[m], flush=True)
E = {"best": best, "tune": {m: {str(k): v for k, v in t.items()} for m, t in tune.items()}, "snrs": SNRS}
for m in ["M1", "M2", "M3", "OPT", "C22"]:
    E[m] = dict(acc=[], acc_std=[], mse=[], rx=[])
    for s in SNRS:
        r = S.avg_over_seeds(m, s, thr=best[m])
        E[m]["acc"].append(float(np.mean(r["acc"][-10:]))); E[m]["acc_std"].append(float(np.std(r["acc_final_all"])))
        E[m]["mse"].append(float(np.mean(r["mse"]))); E[m]["rx"].append(float(np.nanmean(r["rx_snr_db"])))
    print(m, "acc", np.round(E[m]["acc"], 3), "mse", np.round(E[m]["mse"], 3), "rx", np.round(E[m]["rx"], 1), flush=True)
    json.dump(E, open("results_norm.json", "w"), indent=1)
