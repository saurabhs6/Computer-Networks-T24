"""[4] (Y4) runs: tuning on rounds 0-99, then the test runs (selections come from y4cache/)."""
import os, sys, runner as R
import exp_main as EM
stage = sys.argv[1]
if stage == "tune":
    J = [dict(method="Y4", snr=EM.REF, seed=s, thr=t, block="train") for s in range(5) for t in (0.7, 0.9, 1.1)]
    R.run_jobs(J, "v3/y4tune.jsonl", 1)
else:
    t = float(os.environ["Y4_THR"]); t8 = float(os.environ.get("N8_THR", "0"))
    J = [dict(method="Y4", snr=snr, seed=s, thr=t, curve=snr in EM.CURVE_SNRS) for snr in EM.SNRS for s in EM.SEEDS]
    J += [dict(method="Y4", snr=EM.REF, seed=s, thr=t, eps=e) for e in (0.2, 0.4) for s in EM.SEEDS]
    if t8:
        J += [dict(method="Y4", snr=snr, seed=s, thr=t8, n_rx=8) for snr in (-40, -35, -25) for s in EM.SEEDS]
    R.run_jobs(J, "v3/y4.jsonl", int(sys.argv[2]) if len(sys.argv) > 2 else 1)
