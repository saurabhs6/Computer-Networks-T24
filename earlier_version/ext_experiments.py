"""Extra experiments for the corrections round: realised receive SNR, M2+fixed combiner,
[7]-structure baseline (OPT), and the threshold design rule for K = 20, 50, 100."""
import json, numpy as np
import ota_fl_sim as S

R = json.load(open("results.json"))
TH = R["best_thr"]
SNRS = R["snr"]["snrs"]
cfg = {"M1": 0.0, "M2": TH["M2"], "M3": TH["M3"], "M1-BF": 0.0, "M2-EIG": TH["M3"], "OPT": 0.0}

E = {"snrs": SNRS, "cfg": cfg}
for m, th in cfg.items():
    E[m] = dict(acc=[], acc_std=[], mse=[], part=[], rx=[])
    for s in SNRS:
        r = S.avg_over_seeds(m, s, thr=th)
        E[m]["acc"].append(float(np.mean(r["acc"][-10:])))
        E[m]["acc_std"].append(float(np.std(r["acc_final_all"])))
        E[m]["mse"].append(float(np.mean(r["mse"])))
        E[m]["part"].append(float(np.mean(r["part"])))
        E[m]["rx"].append(float(np.nanmean(r["rx_snr_db"])))
    print(m, "acc", np.round(E[m]["acc"], 3), "rx", np.round(E[m]["rx"], 1))

EPS = R["csi"]["eps"]
E["csi"] = {"eps": EPS}
for m in ("M2-EIG", "OPT"):
    E["csi"][m] = dict(acc=[], mse=[])
    for e in EPS:
        r = S.avg_over_seeds(m, S.SNR_REF, thr=cfg[m], eps=e)
        E["csi"][m]["acc"].append(float(np.mean(r["acc"][-10:])))
        E["csi"][m]["mse"].append(float(np.mean(r["mse"])))
    print(m, "csi acc", np.round(E["csi"][m]["acc"], 3))

# ---- threshold design rule -------------------------------------------------
def rule(Kdev):
    h = np.linspace(0.05, 2.0, 2000)
    n = Kdev * np.exp(-h ** 2)
    eta_proxy = h ** 2 + 1.0 / np.maximum(n, 1e-9)       # E[min gain | participating]
    mse = 1.0 / (n ** 2 * eta_proxy)
    ok = n >= 1
    return float(h[ok][np.argmin(mse[ok])])

E["rule"] = {}
GRID = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
for Kdev in (20, 50, 100):
    S.K = Kdev
    mse, acc, part = [], [], []
    for th in GRID:
        r = S.avg_over_seeds("M2", S.SNR_REF, thr=th)
        mse.append(float(np.mean(r["mse"]))); acc.append(float(np.mean(r["acc"][-10:])))
        part.append(float(np.mean(r["part"])))
    i = int(np.argmin(mse))
    E["rule"][str(Kdev)] = dict(grid=GRID, mse=mse, acc=acc, part=part,
                                emp_mse_opt=GRID[i], emp_acc_opt=GRID[int(np.argmax(acc))],
                                theory=rule(Kdev))
    print("K", Kdev, "theory", E["rule"][str(Kdev)]["theory"], "emp mse-opt", GRID[i],
          "acc-opt", E["rule"][str(Kdev)]["emp_acc_opt"], "mse", np.round(mse, 3))
S.K = 50
json.dump(E, open("results_ext.json", "w"), indent=1)
