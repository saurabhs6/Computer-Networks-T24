"""Experiments for the minor corrections (digits task, unit-power symbols).
  lrtune : server learning rate tuned per policy on the tuning rounds (0-99), five seeds, -35 dB
  lrrun  : every policy at its own tuned learning rate, reporting rounds, ten seeds, -45 to -25 dB
  rhomax : common scaling s_k = g_k / rho_max instead of g_k / rho_k (removes the bias of the mean-rho rescaling)
  y14cap : Y14 and Y14-U with the per-round peak limit
  pilot  : U variants whose amplitude sum is measured with one noisy pilot symbol, with and without CSI error
Run:  python exp_minor.py <stage> [processes]"""
import json, os, sys
import numpy as np
import runner as R
from exp_main import P, REF, SEEDS

LR_GRID = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
SNRS = [-45, -40, -35, -30, -25]
# (label, method, debias)
POL = [("M1", "M1", False), ("M2", "M2", False), ("M3", "M3", False), ("M2-EIG", "M2-EIG", False), ("M1-BF", "M1-BF", False),
       ("OPT", "OPT", False), ("OPT-U", "OPT", True), ("RCI", "RCI", False), ("RCI-U", "RCI", True), ("C22", "C22", False),
       ("Z10", "Z10", False), ("Z10-U", "Z10", True), ("Y14", "Y14", False), ("Y14-U", "Y14", True)]


def job(m, deb, **kw):
    return dict(method=m, debias=deb, **P[m], **kw)


def jobs_lrtune():
    J = [dict(method="M1", snr=REF, seed=s, thr=0.0, ideal=True, block="train", lr=lr) for lr in LR_GRID for s in range(5)]
    # M1 kept improving towards small rates, so its grid goes two steps lower
    J += [dict(method="M1", snr=REF, seed=s, thr=0.0, block="train", lr=lr, debias=False) for lr in (0.0625, 0.125) for s in range(5)]
    for lab, m, deb in POL:
        J += [job(m, deb, snr=REF, seed=s, block="train", lr=lr) for lr in LR_GRID for s in range(5)]
    return J


def tuned_lr():
    """learning rate with the highest mean accuracy (last ten rounds) on the tuning rounds, per policy label"""
    acc = {}
    for line in open("v3/lrtune.jsonl"):
        r = json.loads(line); j = r["job"]
        lab = "IDEAL" if j.get("ideal") else j["method"] + ("-U" if j.get("debias") else "")
        acc.setdefault(lab, {}).setdefault(j["lr"], []).append(r["acc10"])
    return {lab: max(d, key=lambda lr: (np.mean(d[lr]), -lr)) for lab, d in acc.items()}


def jobs_lrrun():
    T = tuned_lr()
    J = [dict(method="M1", snr=REF, seed=s, thr=0.0, ideal=True, lr=T["IDEAL"], curve=True) for s in SEEDS]
    for lab, m, deb in POL:
        J += [job(m, deb, snr=snr, seed=s, lr=T[lab], curve=(snr == REF)) for snr in SNRS for s in SEEDS]
    return J


def jobs_rhomax():
    J = [dict(method="M1", snr=REF, seed=s, thr=0.0, ideal=True, scaling="max") for s in SEEDS]
    for lab, m, deb in POL:
        J += [job(m, deb, snr=snr, seed=s, scaling="max") for snr in (-40, -35, -25) for s in SEEDS]
    return J


def jobs_y14cap():
    return [job("Y14", deb, snr=snr, seed=s, y14cap=True) for deb in (False, True) for snr in SNRS for s in SEEDS]


def jobs_pilot():
    J = []
    for m in ("OPT", "RCI", "Z10", "Y14"):
        J += [job(m, True, snr=snr, seed=s, pilot=True) for snr in SNRS for s in SEEDS]
        J += [job(m, True, snr=REF, seed=s, pilot=True, eps=0.2) for s in SEEDS]
    return J


if __name__ == "__main__":
    which = sys.argv[1]
    J = {"lrtune": jobs_lrtune, "lrrun": jobs_lrrun, "rhomax": jobs_rhomax, "y14cap": jobs_y14cap, "pilot": jobs_pilot}[which]()
    if which == "lrrun":
        print("tuned learning rates:", tuned_lr(), flush=True)
    R.run_jobs(J, f"v3/{which}.jsonl", int(sys.argv[2]) if len(sys.argv) > 2 else 1)
