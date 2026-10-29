import json, numpy as np, mnist_cnn_sim as M
import ota_fl_sim as S
S.C22_KAPPA = 10.0
M.LR_C = 0.3; RHO = 0.0336
data = M.load(); out = {}
for tau in (0.5, 0.7):
    r = M.run("C22", -30, 9, tau, data, RHO)
    out[tau] = float(np.mean(r["acc"][-3:])); print(tau, out[tau], flush=True)
    json.dump(out, open("mnist_c22_tune2.json", "w"))
