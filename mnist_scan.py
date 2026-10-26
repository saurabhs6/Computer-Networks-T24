import json, numpy as np, mnist_cnn_sim as M
from multiprocessing import Pool
M.LR_C = 0.3; RHO = 0.0336
data = None
def job(a):
    global data
    if data is None: data = M.load()
    m, snr, thr, seed = a
    r = M.run(m, snr, seed, thr, data, RHO)
    return (m, snr, seed, r)
if __name__ == "__main__":
    jobs = [(m, s, th, 0) for s in (-30, -25, -20, -15) for m, th in (("M1", 0.0), ("M2", 0.7))]
    with Pool(2) as p: out = p.map(job, jobs)
    res = [dict(method=m, snr=s, seed=sd, acc=r["acc"], mse=r["mse"], part=r["part"], rx=r["rx"]) for m, s, sd, r in out]
    json.dump(res, open("mnist_scan.json", "w"))
    for r in res: print(r["method"], r["snr"], "acc last3", round(float(np.mean(r["acc"][-3:])), 3), "mse", round(r["mse"], 2), "part", round(r["part"], 2), "rx", round(r["rx"], 1))
