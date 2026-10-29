"""Regularised channel inversion (RCI), the form reported in [12], as an extra baseline on the digits task."""
import json, numpy as np, ota_fl_sim as S
R = json.load(open("results.json")); SNRS = R["snr"]["snrs"]
LAMS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
tune = {}
for lam in LAMS:
    r = S.avg_over_seeds("RCI", S.SNR_REF, thr=lam, seeds=S.SEEDS[:5], block=S.TRAIN_SLICE)
    tune[lam] = float(np.mean(r["acc"][-10:]))
best = max(tune, key=tune.get); print("RCI tuning", {k: round(v, 3) for k, v in tune.items()}, "best", best)
E = {"lam": best, "tune": {str(k): v for k, v in tune.items()}, "acc": [], "acc_std": [], "mse": [], "rx": [], "part": []}
for s in SNRS:
    r = S.avg_over_seeds("RCI", s, thr=best)
    E["acc"].append(float(np.mean(r["acc"][-10:]))); E["acc_std"].append(float(np.std(r["acc_final_all"])))
    E["mse"].append(float(np.mean(r["mse"]))); E["rx"].append(float(np.nanmean(r["rx_snr_db"]))); E["part"].append(1.0)
print("acc", np.round(E["acc"], 3)); print("mse", np.round(E["mse"], 3)); print("rx", np.round(E["rx"], 1))
E["csi"] = {"eps": R["csi"]["eps"], "acc": [], "mse": []}
for e in R["csi"]["eps"]:
    r = S.avg_over_seeds("RCI", S.SNR_REF, thr=best, eps=e)
    E["csi"]["acc"].append(float(np.mean(r["acc"][-10:]))); E["csi"]["mse"].append(float(np.mean(r["mse"])))
print("csi acc", np.round(E["csi"]["acc"], 3))
json.dump(E, open("results_ext2.json", "w"), indent=1)
