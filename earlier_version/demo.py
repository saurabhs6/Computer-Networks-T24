"""Quick live demo (about 1 minute): M1, M2, M3 at the reference SNR of -25 dB, 3 seeds.
Run:  python demo.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import ota_fl_sim as S

SEEDS = [0, 1, 2]
print(f"{'Method':<8}{'Accuracy %':>12}{'Agg. MSE':>12}{'Participation':>15}{'Rx SNR dB':>11}")
ideal = S.avg_over_seeds("M1", S.SNR_REF, ideal=True, seeds=SEEDS)
print(f"{'Ideal':<8}{100*np.mean(ideal['acc'][-10:]):>12.1f}{'-':>12}{'-':>15}{'-':>11}")
for m, th in [("M1", 0.0), ("M2", 0.5), ("M3", 0.7)]:
    r = S.avg_over_seeds(m, S.SNR_REF, thr=th, seeds=SEEDS)
    print(f"{m:<8}{100*np.mean(r['acc'][-10:]):>12.1f}{np.mean(r['mse']):>12.3f}"
          f"{np.mean(r['part']):>15.2f}{np.nanmean(r['rx_snr_db']):>11.1f}")
