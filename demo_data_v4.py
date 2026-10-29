"""Exports the data of the revised study for the web demo (same numbers as the manuscript)."""
import json, os, sys
import numpy as np
import num_v3 as NV

M = NV.main
IDS = {"M1": "M1", "M2": "M2", "M3": "M3", "M1-BF": "M1-BF", "M2-EIG": "M2-EIG", "Y4": "Y4", "OPT": "OPT", "OPT-U": "OPT-U",
       "RCI": "RCI", "RCI-U": "RCI-U", "C12": "C22", "Z10": "Z10", "Z10-U": "Z10-U", "Y14": "Y14", "Y14-U": "Y14-U"}
SNRS = sorted(s for (m, s) in M if m == "M2" and isinstance(s, (int, float)))
f = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 6)
E = {"snrs": SNRS}
for k, m in IDS.items():
    row = dict(acc=[], acc_std=[], mse=[], mse_med=[], part=[], rx=[], proj=[], power=[])
    for s in SNRS:
        a, sd = NV.A(M, (m, s))
        row["acc"].append(f(a / 100)); row["acc_std"].append(f(sd / 100))
        for fld, key in (("mse", "mse_mean"), ("mse_med", "mse_med"), ("part", "part"), ("rx", "rx"), ("proj", "proj"), ("power", "power")):
            row[fld].append(f(NV.F(M, (m, s), key)))
    E[k] = row
rule = {}
for r in NV.rule_rows():
    Kd = r["K"]; ths = sorted(t for (k, t) in NV.rule if k == Kd)
    rule[str(Kd)] = dict(grid=ths, mse=[f(NV.F(NV.rule, (Kd, t), "mse_mean")) for t in ths],
                         acc=[f(NV.A(NV.rule, (Kd, t))[0] / 100) for t in ths],
                         part=[f(NV.F(NV.rule, (Kd, t), "part")) for t in ths], theory=r["rule"], emp_mse_opt=r["mse_opt"])
E["rule"] = rule
curve = {"IDEAL": [f(x) for x in np.mean([r["curve"] for r in M[("IDEAL", NV.REF)].values()], 0)]}
for k, m in IDS.items():
    rows = M.get((m, NV.REF), {})
    if not rows or "curve" not in next(iter(rows.values())):
        rows = M[(m, "curve")]
    curve[k] = [f(x) for x in np.mean([r["curve"] for r in rows.values()], 0)]
EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
csi = {"eps": EPS}
for k, m in IDS.items():
    es = [e for e in EPS if (m, e) in NV.csi]
    if len(es) < 2:
        continue
    csi[k] = dict(eps=es, acc=[f(NV.A(NV.csi, (m, e))[0] / 100) for e in es],
                  mse=[f(NV.F(NV.csi, (m, e), "mse_mean")) for e in es])
R = {"snr": {"IDEAL_acc": f(NV.IDEAL / 100)}, "curve": curve, "csi": csi, "ref": NV.REF}


def big(g, order, sweep_ids, sweep_snrs):
    rows = {}
    for k in order:
        m = "IDEAL" if k == "IDEAL" else IDS[k]
        key = (m, -35.0)
        if key not in g or not g[key]:
            continue
        a, sd = NV.bacc(g, key)
        rows[k] = dict(acc=f(a), std=f(sd), n=len(g[key]),
                       mse=None if k == "IDEAL" else f(NV.bP(g, m, "mse_mean")),
                       proj=1.0 if k == "IDEAL" else f(NV.bP(g, m, "proj")),
                       part=1.0 if k == "IDEAL" else f(NV.bP(g, m, "part")))
    sw = []
    for s in sweep_snrs:
        row = {"snr": s}
        for k in sweep_ids:
            key = (IDS[k], float(s))
            if key in g and g[key]:
                a, sd = NV.bacc(g, key); row[k] = f(a); row[k + "_sd"] = f(sd)
        sw.append(row)
    return dict(rows=rows, sweep=sw)


MN = big(NV.MN, ["IDEAL", "M1", "M2", "M3", "OPT", "OPT-U", "RCI", "C12", "Z10", "Z10-U", "Y14", "Y14-U"],
         ["M1", "M2", "M3", "C12", "OPT-U"], [-45, -40, -30])
CF = big(NV.CF, ["IDEAL", "M1", "M2", "M3", "OPT", "OPT-U", "C12", "Z10", "Z10-U", "Y14", "Y14-U"],
         ["M2", "M3", "C12", "OPT-U"], [-40, -30])
# fixed scaling (earlier version), full sweeps
R0 = json.load(open(os.path.join(NV.SIM, "results.json"))); e1 = json.load(open(os.path.join(NV.SIM, "results_ext.json")))
e2 = json.load(open(os.path.join(NV.SIM, "results_ext2.json"))); e3 = json.load(open(os.path.join(NV.SIM, "results_ext3.json")))
FX = {"snrs": R0["snr"]["snrs"]}
for k in ("M1", "M2", "M3", "M1-BF"):
    FX[k] = dict(acc=R0["snr"][k]["acc"], mse=R0["snr"][k]["mse"])
for k in ("M2-EIG", "OPT"):
    FX[k] = dict(acc=e1[k]["acc"], mse=e1[k]["mse"])
FX["RCI"] = dict(acc=e2["acc"], mse=e2["mse"]); FX["C12"] = dict(acc=e3["acc"], mse=e3["mse"])
D = dict(E=E, R=R, MN=MN, CF=CF, FX=FX)
out = sys.argv[1] if len(sys.argv) > 1 else "demo_data_v4.json"
json.dump(D, open(out, "w"))
print("written", out, os.path.getsize(out), "bytes")
