"""Main digits experiments of the revised study (unit-power symbols, reference SNR -35 dB)."""
import sys, runner as R
REF = -35
SNRS = [-45, -40, -35, -30, -25, -20, -15, -10, 0, 10, 20]
SEEDS = range(10)
P = {"M1": dict(thr=0.0), "M2": dict(thr=0.7), "M3": dict(thr=0.7), "M2-EIG": dict(thr=0.7), "M1-BF": dict(thr=0.0),
     "OPT": dict(thr=0.0), "RCI": dict(thr=0.2), "C22": dict(thr=0.8, kappa=100.0), "Z10": dict(thr=0.0), "Y14": dict(thr=0.0)}
CURVE_SNRS = (-40, -35, -25)

def jobs_main():
    J = [dict(method="M1", snr=REF, seed=s, thr=0.0, ideal=True, curve=True) for s in SEEDS]
    for m, kw in P.items():
        for snr in SNRS:
            J += [dict(method=m, snr=snr, seed=s, curve=snr in CURVE_SNRS, **kw) for s in SEEDS]
    return J

def jobs_csi(methods=P):
    J = []
    for m in methods:
        for e in (0.05, 0.1, 0.2, 0.3, 0.4):
            J += [dict(method=m, snr=REF, seed=s, eps=e, **P[m]) for s in SEEDS]
    return J

def jobs_thr():
    return [dict(method="M2", snr=REF, seed=s, thr=t) for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2) for s in SEEDS]

def jobs_rule():
    return [dict(method="M2", snr=REF, seed=s, thr=t, K=k) for k in (20, 50, 100)
            for t in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5) for s in SEEDS]

def jobs_jakes():
    J = [dict(method="M1", snr=REF, seed=s, thr=0.0, ideal=True, fd=fd) for fd in (0.01, 0.05) for s in SEEDS]
    for fd in (0.01, 0.05):
        for m in P:
            J += [dict(method=m, snr=REF, seed=s, fd=fd, **P[m]) for s in SEEDS]
    return J

def jobs_alpha():
    J = [dict(method="M1", snr=REF, seed=s, thr=0.0, ideal=True, alpha=a) for a in (0.1, 100.0) for s in SEEDS]
    for a in (0.1, 100.0):
        for m in P:
            J += [dict(method=m, snr=REF, seed=s, alpha=a, **P[m]) for s in SEEDS]
    return J

def jobs_n8_tune():
    return [dict(method=m, snr=REF, seed=s, thr=t, n_rx=8, block="train") for m in ("M3",)
            for t in (0.4, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3, 1.5) for s in range(5)]

N8_THR = None   # set after tuning

def jobs_n8():
    import os
    t = float(os.environ["N8_THR"])
    J = []
    for snr in (-40, -35, -25):
        for m, th in (("M3", t), ("M2-EIG", t), ("M1-BF", 0.0)):
            J += [dict(method=m, snr=snr, seed=s, thr=th, n_rx=8) for s in SEEDS]
    return J

def jobs_debias():
    J = []
    for m in ("OPT", "RCI", "Z10", "Y14"):
        for snr in (-45, -40, -35, -30, -25):
            J += [dict(method=m, snr=snr, seed=s, debias=True, **P[m]) for s in SEEDS]
    for m in ("M1", "M2", "M3", "M2-EIG", "M1-BF", "C22", "OPT", "RCI", "Z10", "Y14"):   # direction metric (cos) at three SNRs
        for snr in (-40, -35, -25):
            J += [dict(method=m, snr=snr, seed=s, tag="cos", **P[m]) for s in SEEDS]
    return J

def jobs_uvar():
    """unit-gain (debiased) variants of the MSE-optimised policies in the robustness settings"""
    J = []
    for m in ("OPT", "RCI", "Z10", "Y14"):
        kw = dict(P[m], debias=True)
        J += [dict(method=m, snr=REF, seed=s, eps=e, **kw) for e in (0.05, 0.1, 0.2, 0.3, 0.4) for s in SEEDS]
        J += [dict(method=m, snr=REF, seed=s, fd=fd, **kw) for fd in (0.01, 0.05) for s in SEEDS]
        J += [dict(method=m, snr=REF, seed=s, alpha=a, **kw) for a in (0.1, 100.0) for s in SEEDS]
        J += [dict(method=m, snr=snr, seed=s, **kw) for snr in (-20, -15, -10, 0, 10, 20) for s in SEEDS]
        J += [dict(method=m, snr=REF, seed=s, curve=True, tag="curve", **kw) for s in SEEDS]
    return J

if __name__ == "__main__":
    which = sys.argv[1]
    J = {"main": jobs_main, "csi": jobs_csi, "thr": jobs_thr, "rule": jobs_rule, "jakes": jobs_jakes,
         "alpha": jobs_alpha, "n8tune": jobs_n8_tune, "debias": jobs_debias, "n8": jobs_n8, "uvar": jobs_uvar}[which]()
    R.run_jobs(J, f"v3/{which}.jsonl", int(sys.argv[2]) if len(sys.argv) > 2 else 1)
