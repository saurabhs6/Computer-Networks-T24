"""Pre-compute the [4] selections (independent of the noise level). Usage: python y4_pre.py <job-list name>"""
import sys, itertools, numpy as np, ota_fl_sim as S
from multiprocessing import Pool

def job(a):
    seed, thr, eps, blk, n_rx, fd, alpha = a
    S.N_RX, S.JAKES_FD, S.DIRICHLET_ALPHA = n_rx, fd, alpha
    S.y4_trace(seed, thr, eps, S.TRAIN_SLICE if blk == 0 else S.TEST_SLICE)
    return a

LISTS = {
    "tune": [(s, t, 0.0, 0, 4, 0.0, 0.5) for s in range(5) for t in (0.7, 0.9, 1.1)],
}
def test_list(thr, thr8):
    J = [(s, thr, e, 1, 4, 0.0, 0.5) for e in (0.0, 0.2, 0.4) for s in range(10)]
    if thr8:
        J += [(s, thr8, 0.0, 1, 8, 0.0, 0.5) for s in range(10)]
    return J

if __name__ == "__main__":
    if sys.argv[1] == "test":
        import os
        jobs = test_list(float(os.environ["Y4_THR"]), float(os.environ.get("N8_THR", "0")))
    else:
        jobs = LISTS[sys.argv[1]] if sys.argv[1] in LISTS else eval(sys.argv[1])
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    with Pool(nproc) as p:
        for a in p.imap_unordered(job, jobs):
            print("done", a, flush=True)
