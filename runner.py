"""Generic resumable job runner for the digits experiments of the revised study (unit-power symbols).
A job is a dict; results are appended to a JSON-lines file, one line per (configuration, seed)."""
import json, os, sys, warnings
import numpy as np
import ota_fl_sim as S

DEFAULT = dict(norm=True, kappa=10.0, eps=0.0, block="test", n_rx=4, fd=0.0, alpha=0.5, K=50, curve=False, ideal=False,
               debias=False)
EXTRA = dict(lr=1.0, scaling="mean", y14cap=False, pilot=False)   # options added for the minor corrections




def key(j):
    j = {**DEFAULT, **j}
    j = {k: v for k, v in j.items() if not (k in EXTRA and v == EXTRA[k])}   # old result files keep their keys
    return json.dumps({k: j[k] for k in sorted(j) if k not in ("curve",)}, sort_keys=True)


def summarise(log, curve):
    a = np.array(log["acc"]); m = np.array(log["mse"]); se = np.array(log["sqrt_eta"], float)
    out = dict(acc10=float(a[-10:].mean()), acc_last=float(a[-1]),
               mse_mean=float(m.mean()), mse_med=float(np.median(m)), mse_p90=float(np.percentile(m, 90)),
               mse_p99=float(np.percentile(m, 99)), mse_max=float(m.max()),
               part=float(np.mean(log["part"])), power=float(np.mean(log["power"])), peak=float(np.max(log["peak"])),
               noise=float(np.mean(log["noise_var"])), rx=float(np.nanmean(log["rx_snr_db"])) if np.any(np.isfinite(log["rx_snr_db"])) else None,
               proj=float(np.mean(log["proj"])), proj_med=float(np.median(log["proj"])),
               cos=float(np.mean(log["cos"])) if log.get("cos") else None)
    fin = se[np.isfinite(se)]
    if len(fin):
        eta = fin ** 2
        out.update(eta_med=float(np.median(eta)), eta_p10=float(np.percentile(eta, 10)), eta_p1=float(np.percentile(eta, 1)),
                   eta_min=float(eta.min()))
    a90 = np.where(a >= 0.9)[0]
    out["r90"] = int(a90[0]) + 1 if len(a90) else None
    if curve:
        out["curve"] = [round(float(x), 5) for x in a]
        out["mse_curve"] = [float(x) for x in m]
        out["eta_curve"] = [float(x) for x in se]
        out["proj_curve"] = [float(x) for x in log["proj"]]
    return out


def work(j):
    warnings.filterwarnings("ignore")
    j = {**DEFAULT, **j}
    S.NORM, S.C22_KAPPA, S.N_RX, S.JAKES_FD, S.DIRICHLET_ALPHA, S.K = j["norm"], j["kappa"], j["n_rx"], j["fd"], j["alpha"], j["K"]
    S.DEBIAS = j["debias"]
    x = {**EXTRA, **j}
    S.LR, S.SCALING, S.Y14_CAP, S.PILOT = x["lr"], x["scaling"], x["y14cap"], x["pilot"]
    blk = S.TRAIN_SLICE if j["block"] == "train" else S.TEST_SLICE
    log = S.run(j["method"], j["snr"], j["seed"], j["thr"], j["eps"], block=blk, ideal=j["ideal"])
    return dict(job=j, **summarise(log, j["curve"]))


def run_jobs(jobs, out_path, nproc=1):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done.add(key(json.loads(line)["job"]))
    todo = [j for j in jobs if key(j) not in done]
    print(len(jobs), "jobs,", len(todo), "to do", flush=True)
    if not todo:
        return
    if nproc == 1:
        it = map(work, todo)
    else:
        from multiprocessing import Pool
        pool = Pool(nproc); it = pool.imap_unordered(work, todo, chunksize=1)
    with open(out_path, "a") as f:
        for i, r in enumerate(it):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 20 == 0:
                print(i, r["job"]["method"], r["job"]["snr"], round(r["acc10"], 4), flush=True)


def load(out_path, **flt):
    rows = []
    for line in open(out_path):
        r = json.loads(line)
        if all(r["job"].get(k, DEFAULT.get(k)) == v for k, v in flt.items()):
            rows.append(r)
    return rows
