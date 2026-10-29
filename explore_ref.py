"""Choice of the reference SNR on the tuning rounds only (unit-power symbols)."""
import json, numpy as np, ota_fl_sim as S
S.NORM = True; S.C22_KAPPA = 10.0
out = {}
for snr in (-45, -40, -35, -30, -25):
    out[snr] = {}
    for m, th in [("M1",0),("M2",0.5),("M3",0.7),("C22",0.8),("OPT",0),("Z10",0),("Y14",0),("RCI",0.05)]:
        r = S.avg_over_seeds(m, snr, thr=th, seeds=S.SEEDS[:5], block=S.TRAIN_SLICE)
        out[snr][m] = float(np.mean(r["acc"][-10:]))
    print(snr, {k: round(v, 3) for k, v in out[snr].items()}, flush=True)
json.dump(out, open("v3/explore_ref.json", "w"), indent=1)
