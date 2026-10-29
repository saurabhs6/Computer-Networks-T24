"""CIFAR-10 runs of the revised study (resumable, v3/cifar_v3.jsonl).
Stages: lr   - learning rate of the noiseless reference on tuning seed 9
        tune - C12 target and a quick SNR check on seed 9
        main - 5 seeds x 3 SNRs x policies"""
import json, os, sys, time
import numpy as np, torch
import cifar_sim as C
OUT = "v3/cifar_v3.jsonl"
LR = float(os.environ.get("CIFAR_LR", "0.1"))
TAU = float(os.environ.get("C22_TAU", "0.8"))
SNRS = [float(x) for x in os.environ.get("CIFAR_SNRS", "-40,-35,-30").split(",")]
cfg = {"M1": 0.0, "M2": 0.7, "M3": 0.7, "OPT": 0.0, "C22": TAU, "Z10": 0.0, "Y14": 0.0, "OPT-U": 0.0, "Z10-U": 0.0, "Y14-U": 0.0}

def jobs(stage):
    if stage == "lr":
        return [("IDEAL", 9, -35.0, 0.0, lr) for lr in (0.05, 0.1, 0.2, 0.4)]
    if stage == "tune":
        return [("C22", 9, -35.0, t, LR) for t in (0.5, 0.8, 1.2)] + [("M2", 9, -35.0, 0.7, LR), ("M1", 9, -35.0, 0.0, LR)] \
            + [("C22", 9, -35.0, 2.0, LR)]
    J = [("IDEAL", s, -35.0, 0.0, LR) for s in range(5)]
    for snr in SNRS:                      # all policies at the reference SNR, the main ones at the other two
        for m in (cfg if snr == -35.0 else ("M2", "M3", "C22", "OPT-U")):
            J += [(m, s, snr, cfg[m], LR) for s in range(5)]
    return J

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            r = json.loads(l); done.add((r["method"], r["seed"], r["snr"], r["thr"], r["lr"]))
    J = jobs(sys.argv[1])
    if len(sys.argv) > 3:      # shard i of n (of the full list, so shards started at different times stay disjoint)
        i, n = int(sys.argv[2]), int(sys.argv[3]); J = J[i::n]
    J = [j for j in J if j not in done]
    print(len(J), "jobs", flush=True)
    data = C.load()
    for (m, s, snr, th, lr) in J:
        t = time.time()
        C.S.DEBIAS = m.endswith("-U")
        base = m[:-2] if C.S.DEBIAS else m
        r = C.run("M1" if m == "IDEAL" else base, snr, s, th, data, lr, ideal=(m == "IDEAL"))
        out = dict(method=m, seed=s, snr=snr, thr=th, lr=lr, secs=round(time.time() - t, 1), **r)
        with open(OUT, "a") as f: f.write(json.dumps(out) + "\n")
        print(m, s, snr, th, lr, [round(a, 3) for a in r["acc"][-3:]], out["secs"], "s", flush=True)
