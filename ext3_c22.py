"""[12] Case I policy (max-power constraint) on the digits task: tune tau, kappa on tuning rounds, sweep SNR and CSI."""
import json, itertools, numpy as np, ota_fl_sim as S
R = json.load(open("results.json")); SNRS = R["snr"]["snrs"]
TAUS = [0.5, 0.8, 1.2, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]; KAPS = [0.1, 1.0, 10.0]
tune = {}
for kap, tau in itertools.product(KAPS, TAUS):
    S.C22_KAPPA = kap
    r = S.avg_over_seeds("C22", S.SNR_REF, thr=tau, seeds=S.SEEDS[:5], block=S.TRAIN_SLICE)
    tune[(kap, tau)] = float(np.mean(r["acc"][-10:]))
    print(kap, tau, round(tune[(kap, tau)], 3), flush=True)
kap, tau = max(tune, key=tune.get); S.C22_KAPPA = kap
print("best", kap, tau, tune[(kap, tau)])
E = {"kappa": kap, "tau": tau, "tune": {f"{k[0]}|{k[1]}": v for k, v in tune.items()},
     "acc": [], "acc_std": [], "mse": [], "rx": [], "power": []}
for s in SNRS:
    r = S.avg_over_seeds("C22", s, thr=tau)
    E["acc"].append(float(np.mean(r["acc"][-10:]))); E["acc_std"].append(float(np.std(r["acc_final_all"])))
    E["mse"].append(float(np.mean(r["mse"]))); E["rx"].append(float(np.nanmean(r["rx_snr_db"])))
    E["power"].append(float(np.mean(r["power"])))
print("acc", np.round(E["acc"], 3)); print("pow", np.round(E["power"], 3))
E["csi"] = {"eps": R["csi"]["eps"], "acc": [], "mse": []}
for e in R["csi"]["eps"]:
    r = S.avg_over_seeds("C22", S.SNR_REF, thr=tau, eps=e)
    E["csi"]["acc"].append(float(np.mean(r["acc"][-10:]))); E["csi"]["mse"].append(float(np.mean(r["mse"])))
print("csi", np.round(E["csi"]["acc"], 3))
json.dump(E, open("results_ext3.json", "w"), indent=1)
