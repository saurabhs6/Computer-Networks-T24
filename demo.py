"""Quick live demo (one to two minutes): the main policies at the reference SNR of -35 dB with unit-power symbols, 3 seeds.
Run:  python demo.py"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import ota_fl_sim as S

S.NORM = True                     # unit-power symbols, as in the revised paper
SNR, SEEDS = -35, [0, 1, 2]
POLICIES = [("M1", 0.0, 10.0, False), ("M2", 0.7, 10.0, False), ("M3", 0.7, 10.0, False), ("C22", 0.8, 100.0, False),
            ("OPT", 0.0, 10.0, False), ("OPT", 0.0, 10.0, True), ("Z10", 0.0, 10.0, False), ("Z10", 0.0, 10.0, True)]
print(f"{'Policy':<8}{'Accuracy %':>12}{'Agg. MSE':>12}{'Shrinkage':>11}{'Participation':>15}")
ideal = S.avg_over_seeds("M1", SNR, ideal=True, seeds=SEEDS)
print(f"{'Ideal':<8}{100 * np.mean(ideal['acc'][-10:]):>12.1f}{'-':>12}{'-':>11}{'-':>15}")
for m, th, kap, deb in POLICIES:
    S.C22_KAPPA, S.DEBIAS = kap, deb
    r = S.avg_over_seeds(m, SNR, thr=th, seeds=SEEDS)
    name = {"C22": "C12"}.get(m, m) + ("-U" if deb else "")
    print(f"{name:<8}{100 * np.mean(r['acc'][-10:]):>12.1f}{np.mean(r['mse']):>12.3f}{np.mean(r['proj']):>11.2f}"
          f"{np.mean(r['part']):>15.2f}")
S.DEBIAS = False
