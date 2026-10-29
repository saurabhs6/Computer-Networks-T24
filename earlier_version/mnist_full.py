import json, numpy as np, mnist_cnn_sim as M
from multiprocessing import Pool
M.LR_C = 0.3; RHO = 0.0336; SNR = -30
cfg = {"IDEAL": 0.0, "M1": 0.0, "M2": 0.7, "M3": 0.7, "OPT": 0.0, "RCI": 0.05}
data = None
def job(a):
    global data
    if data is None: data = M.load()
    m, seed = a
    r = M.run("M1" if m == "IDEAL" else m, SNR, seed, cfg[m], data, RHO, ideal=(m == "IDEAL"))
    out = dict(method=m, seed=seed, acc=r["acc"], mse=r["mse"], part=r["part"], rx=r["rx"])
    with open("mnist_full_partial.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    return out
if __name__ == "__main__":
    jobs = [(m, s) for s in (0, 1, 2) for m in cfg]
    with Pool(2) as p: res = p.map(job, jobs, chunksize=1)
    json.dump(res, open("mnist_full.json", "w"))
    print("done")
