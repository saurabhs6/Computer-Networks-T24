"""Extra MNIST runs (review item 2): more seeds at -30 dB, [12]-style policy, and a small SNR sweep. Resumable."""
import json, os, numpy as np, mnist_cnn_sim as M
import ota_fl_sim as S
from multiprocessing import Pool
S.C22_KAPPA = 10.0
M.LR_C = 0.3; RHO = 0.0336
TAU = float(os.environ.get("C22_TAU", "0.7"))
cfg = {"IDEAL": 0.0, "M1": 0.0, "M2": 0.7, "M3": 0.7, "OPT": 0.0, "RCI": 0.05, "C22": TAU}
data = None
def job(a):
    global data
    if data is None: data = M.load()
    m, seed, snr = a
    r = M.run("M1" if m == "IDEAL" else m, snr, seed, cfg[m], data, RHO, ideal=(m == "IDEAL"))
    out = dict(method=m, seed=seed, snr=snr, acc=r["acc"], mse=r["mse"], part=r["part"], rx=r["rx"])
    with open("mnist_batch.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    return 1
if __name__ == "__main__":
    done = set()
    if os.path.exists("mnist_batch.jsonl"):
        for l in open("mnist_batch.jsonl"):
            r = json.loads(l); done.add((r["method"], r["seed"], r["snr"]))
    jobs = []
    for s in (0, 1, 2): jobs.append(("C22", s, -30))
    for s in (3, 4):
        for m in ["IDEAL", "M1", "M2", "M3", "OPT", "RCI", "C22"]: jobs.append((m, s, -30))
    for s in (0, 1, 2):
        for snr in (-35, -25, -20):
            for m in ["M1", "M2", "M3", "C22"]: jobs.append((m, s, snr))
    jobs = [j for j in jobs if j not in done]
    print(len(jobs), "jobs", flush=True)
    with Pool(2) as p: p.map(job, jobs, chunksize=1)
    print("done")
