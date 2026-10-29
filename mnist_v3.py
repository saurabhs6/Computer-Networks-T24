"""MNIST runs of the revised study: unit-power symbols, reference SNR -35 dB. Resumable (mnist_v3.jsonl)."""
import json, os, sys, numpy as np, mnist_cnn_sim as M
import ota_fl_sim as S
from multiprocessing import Pool
S.NORM = True; S.C22_KAPPA = 100.0
M.LR_C = 0.3
OUT = "v3/mnist_v3.jsonl"
TAU = float(os.environ.get("C22_TAU", "0.8"))
cfg = {"IDEAL": 0.0, "M1": 0.0, "M2": 0.7, "M3": 0.7, "OPT": 0.0, "RCI": 0.2, "C22": TAU, "Z10": 0.0, "Y14": 0.0,
       "OPT-U": 0.0, "Z10-U": 0.0, "Y14-U": 0.0}
data = None
def job(a):
    global data
    if data is None: data = M.load()
    m, seed, snr, th = a
    S.DEBIAS = m.endswith("-U")           # unit-gain normalisation at the server
    base = m[:-2] if S.DEBIAS else m
    r = M.run("M1" if m == "IDEAL" else base, snr, seed, th, data, 1.0, ideal=(m == "IDEAL"))
    out = dict(method=m, seed=seed, snr=snr, thr=th, **r)
    with open(OUT, "a") as f: f.write(json.dumps(out) + "\n")
    return out
def jobs(stage):
    J = []
    if stage == "tune":        # tuning seed 9 (not reported): C12 target
        J = [("C22", 9, -35, t) for t in (0.5, 0.8, 1.2)] + [("IDEAL", 9, -35, 0.0), ("M2", 9, -35, 0.7)]
    else:
        for s in range(5):
            J += [(m, s, -35, cfg[m]) for m in cfg]
        for s in range(3):
            for snr in (-45, -40, -30):
                J += [(m, s, snr, cfg[m]) for m in ("M1", "M2", "M3", "C22", "OPT-U")]
    return J
if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["method"], r["seed"], r["snr"], r["thr"]))
    J = [j for j in jobs(sys.argv[1]) if j not in done]
    print(len(J), "jobs", flush=True)
    with Pool(int(sys.argv[2])) as p:
        for r in p.imap_unordered(job, J):
            print(r["method"], r["seed"], r["snr"], r["thr"], r["acc"][-3:], flush=True)
