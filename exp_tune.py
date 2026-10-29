"""Tuning on rounds 0-99 (5 seeds) at the reference SNR of -35 dB, unit-power symbols."""
import itertools, runner as R
REF = -35
jobs = []
for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    jobs += [dict(method="M2", snr=REF, seed=s, thr=th, block="train") for s in range(5)]
for th in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2):
    jobs += [dict(method="M3", snr=REF, seed=s, thr=th, block="train") for s in range(5)]
for lam in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
    jobs += [dict(method="RCI", snr=REF, seed=s, thr=lam, block="train") for s in range(5)]
for kap, tau in itertools.product((1.0, 10.0, 100.0), (0.3, 0.5, 0.8, 1.2, 2.0)):
    jobs += [dict(method="C22", snr=REF, seed=s, thr=tau, kappa=kap, block="train") for s in range(5)]
if __name__ == "__main__":
    R.run_jobs(jobs, "v3/tune.jsonl", 1)
