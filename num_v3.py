"""All numbers used in the revised manuscript, read from the result files of sim3/v3.
Importing this module computes N (a dict) and defines the text blocks that depend on results."""
import json, math, os, sys, collections
import numpy as np

SIM = os.environ.get("SIM3", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SIM)
import stats_v3 as T          # noqa: E402

V = lambda f: os.path.join(SIM, "v3", f)
FIGS = {k: os.path.join(SIM, "figs", f"fig_{k}.png") for k in
       ("pipeline", "snr", "curves", "mse", "shrink", "tail", "rxsnr", "threshold", "csi", "cifar")}
NAME = T.NAME
REF = -35
N = {}
pct = lambda x: 100.0 * x


def load(path, by=("method", "snr")):
    return T.group(V(path), by=by)


main = load("main.jsonl")
y4 = load("y4.jsonl", by=("method", "snr", "eps", "n_rx")) if os.path.exists(V("y4.jsonl")) else {}
for (m, snr, eps, nrx), v in y4.items():
    if eps == 0 and nrx == 4:
        main[(m, snr)] = v                        # Y4 rows live in their own file


def accs(g, key):
    return {s: pct(r["acc10"]) for s, r in g[key].items()}


def m_acc(g, key):
    return T.mean_sd(list(accs(g, key).values()))


def m_field(g, key, field, agg=np.mean):
    return float(agg([r[field] for r in g[key].values()]))


# ---------------------------------------------------------------- tuned values
def tuned(path, method, by=("thr",), extra=None):
    g = T.group(V(path), by=("method",) + by)
    best = None
    for k, rows in g.items():
        if k[0] != method:
            continue
        a = np.mean([r["acc10"] for r in rows.values()])
        if best is None or a > best[0]:
            best = (a, k[1:])
    return best[1]


N["thr"] = {"M2": tuned("tune.jsonl", "M2")[0], "M3": tuned("tune.jsonl", "M3")[0], "RCI": tuned("tune.jsonl", "RCI")[0]}
tau, kap = tuned("tune.jsonl", "C22", by=("thr", "kappa"))
N["thr"]["C22"], N["c12_kK"] = tau, kap
N["thr"]["Y4"] = tuned("y4tune.jsonl", "Y4")[0] if os.path.exists(V("y4tune.jsonl")) else 0.9
N["thr"]["N8"] = tuned("n8tune.jsonl", "M3")[0]
N["jakes_r"] = {}
from scipy.special import j0  # noqa: E402
for fd in (0.01, 0.05):
    N["jakes_r"][fd] = float(j0(2 * math.pi * fd))
N["c12_power"] = m_field(main, ("C22", REF), "power")
N["y14_peak"] = m_field(main, ("Y14", REF), "peak", np.max)
N["shrink_hi"] = m_field(main, ("M1", 20), "proj")
K = 50
gain = [2 * (K * r["part"]) ** 2 * r["eta_med"] for r in main[("M2", REF)].values()]
N["coh_gain_db"] = float(10 * np.log10(np.median(gain)))

# ---------------------------------------------------------------- other experiment files
deb = load("debias.jsonl", by=("method", "snr", "debias", "tag"))
uv = load("uvar.jsonl", by=("method", "snr", "eps", "fd", "alpha", "tag")) if os.path.exists(V("uvar.jsonl")) else {}
for (m, snr, db, tag), rows in deb.items():
    if db:
        main[(m + "-U", snr)] = rows
for (m, snr, eps, fd, al, tag), rows in uv.items():
    if eps == 0 and fd == 0 and al == 0.5 and tag is None:
        main[(m + "-U", snr)] = rows
    if tag == "curve":
        main[(m + "-U", "curve")] = rows
csi = load("csi.jsonl", by=("method", "eps"))
for (m, snr, eps, fd, al, tag), rows in uv.items():
    if eps > 0:
        csi[(m + "-U", eps)] = rows
if os.path.exists(V("y4.jsonl")):
    for (m, eps, nrx, snr), rows in load("y4.jsonl", by=("method", "eps", "n_rx", "snr")).items():
        if nrx == 4 and eps > 0 and snr == REF:
            csi[(m, eps)] = rows
for k, rows in list(main.items()):
    if k[1] == REF:
        csi[(k[0], 0.0)] = rows
jk = load("jakes.jsonl", by=("method", "fd"))
al = load("alpha.jsonl", by=("method", "alpha"))
for (m, snr, eps, fd, a_, tag), rows in uv.items():
    if fd > 0:
        jk[(m + "-U", fd)] = rows
    if a_ != 0.5:
        al[(m + "-U", a_)] = rows
thr = load("thr.jsonl", by=("thr",))
rule = load("rule.jsonl", by=("K", "thr"))
n8 = load("n8.jsonl", by=("method", "snr")) if os.path.exists(V("n8.jsonl")) else {}
if os.path.exists(V("y4.jsonl")):
    for (m, nrx, snr), rows in load("y4.jsonl", by=("method", "n_rx", "snr")).items():
        if nrx == 8:
            n8[(m, snr)] = rows


def load_lab(path, extra=()):
    """minor-correction files: {(label, snr, *extra): {seed: row}}, label = method (+ '-U' with unit-gain normalisation)"""
    out = {}
    for (m, snr, db, *x), rows in load(path, by=("method", "snr", "debias") + tuple(extra)).items():
        out[(m + ("-U" if db else ""), snr, *x)] = rows
    return out


lrr = load_lab("lrrun.jsonl")          # per-policy learning rate
rmx = load_lab("rhomax.jsonl")         # common scaling g_k / rho_max
y14c = load_lab("y14cap.jsonl")        # Y14 with the peak limit
plt_ = load_lab("pilot.jsonl", ("eps",))   # U variants with a noisy pilot
csiu = {}
for (m, snr, eps, fd, a_, tag), rows in uv.items():
    if eps > 0:
        csiu[(m + "-U", eps)] = rows
LRT = {}
for line in open(V("lrtune.jsonl")):
    r = json.loads(line); j = r["job"]
    lab = "IDEAL" if j.get("ideal") else j["method"] + ("-U" if j.get("debias") else "")
    LRT.setdefault(lab, {}).setdefault(j["lr"], []).append(100 * r["acc10"])
N["lr"] = {lab: max(d, key=lambda v: (np.mean(d[v]), -v)) for lab, d in LRT.items()}
N["lr_grid"] = sorted({v for d in LRT.values() for v in d})


def A(g, key):
    """(mean, sd) of accuracy in percent"""
    return T.mean_sd([pct(r["acc10"]) for r in g[key].values()])


def F(g, key, field, agg=np.mean):
    return float(agg([r[field] for r in g[key].values()]))


def CI(g1, k1, g2, k2):
    return T.paired_ci(accs(g1, k1), accs(g2, k2))


IDEAL = A(main, ("IDEAL", REF))[0]
N["ideal"] = IDEAL
POL = ["M1", "M2", "M3", "M2-EIG", "M1-BF", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U", "Y4"]
NAME.update({"OPT-U": "OPT-U", "RCI-U": "RCI-U", "Z10-U": "Z10-U", "Y14-U": "Y14-U"})
have = lambda m, s=REF: (m, s) in main


def fmt_ci(c, signed=True):
    s = lambda x: ("0.00" if abs(x) < 0.005 and signed else (f"{x:+.2f}" if signed else f"{x:.2f}")).replace("-", "−")
    star = "" if c["within_noise"] else "*"
    return f"{s(c['diff'])} [{s(c['lo'])}, {s(c['hi'])}]{star}"


# ================================================================ summaries used by the text
def tail_stats(m, snr=REF):
    rows = main.get((m, snr), {})
    if not rows or "mse_curve" not in next(iter(rows.values())):
        rows = main[(m, "curve")]
    mse = np.concatenate([np.array(r["mse_curve"]) for r in rows.values()])
    eta = np.concatenate([np.array(r["eta_curve"]) ** 2 for r in rows.values()])
    eta = eta[np.isfinite(eta)]
    top = np.sort(mse)[-max(1, len(mse) // 100):]
    return dict(mean=float(mse.mean()), med=float(np.median(mse)), p90=float(np.percentile(mse, 90)),
                p99=float(np.percentile(mse, 99)), top1=float(top.sum() / mse.sum()),
                eta_med=float(np.median(eta)), eta_p1=float(np.percentile(eta, 1)))


def seed_rank(m, snr):
    from scipy import stats
    rows = main[(m, snr)]
    acc = [pct(r["acc10"]) for r in rows.values()]
    return dict(mean=float(stats.spearmanr(acc, [r["mse_mean"] for r in rows.values()])[0]),
                med=float(stats.spearmanr(acc, [r["mse_med"] for r in rows.values()])[0]))


def rule_h(Kdev):
    h = np.linspace(0.05, 2.0, 2000)
    n = Kdev * np.exp(-h ** 2)
    mse = 1.0 / (n ** 2 * (h ** 2 + 1.0 / np.maximum(n, 1e-9)))
    ok = n >= 1
    return float(h[ok][np.argmin(mse[ok])])


def rule_rows():
    out = []
    for Kd in (20, 50, 100):
        ths = sorted(t for (k, t) in rule if k == Kd)
        mse = [F(rule, (Kd, t), "mse_mean") for t in ths]
        acc_ = [A(rule, (Kd, t))[0] for t in ths]
        i, j = int(np.argmin(mse)), int(np.argmax(acc_))
        k7 = ths.index(0.7)
        out.append(dict(K=Kd, rule=rule_h(Kd), mse_opt=ths[i], mse_min=mse[i], mse07=mse[k7], gap=100 * (mse[k7] / mse[i] - 1),
                        acc_opt=ths[j], acc_best=acc_[j], acc07=acc_[k7]))
    return out


def old_fixed():
    """fixed scaling rho = 0.1 (earlier version): accuracy at -35/-25 dB and MSE at -25 dB, from the earlier result files"""
    R0 = json.load(open(os.path.join(SIM, "results.json")))
    e1 = json.load(open(os.path.join(SIM, "results_ext.json")))
    e2 = json.load(open(os.path.join(SIM, "results_ext2.json")))
    e3 = json.load(open(os.path.join(SIM, "results_ext3.json")))
    sn = R0["snr"]["snrs"]; a, b = sn.index(-35), sn.index(-25)
    out = {}
    for m in ("M1", "M2", "M3", "M1-BF"):
        out[m] = (pct(R0["snr"][m]["acc"][a]), pct(R0["snr"][m]["acc"][b]), R0["snr"][m]["mse"][b])
    for m in ("M2-EIG", "OPT"):
        out[m] = (pct(e1[m]["acc"][a]), pct(e1[m]["acc"][b]), e1[m]["mse"][b])
    out["RCI"] = (pct(e2["acc"][a]), pct(e2["acc"][b]), e2["mse"][b])
    out["C22"] = (pct(e3["acc"][a]), pct(e3["acc"][b]), e3["mse"][b])
    return out


# ================================================================ formatting helpers
def a1(m, snr=REF, g=None):
    return f"{A(g or main, (m, snr))[0]:.1f}%"


def av(m, snr=REF, g=None):
    return A(g or main, (m, snr))[0]


def sgn(x, k=1):
    if abs(x) < 0.5 * 10 ** -k:
        return f"{0:.{k}f}"
    return f"{x:+.{k}f}".replace("-", "−")


def num(x, k=2):
    return f"{x:.{k}f}".replace("-", "−")


def ci_txt(c):
    """'+0.74 points (95% CI +0.50 to +0.98)'"""
    return f"{sgn(c['diff'], 2)} points (95% CI {sgn(c['lo'], 2)} to {sgn(c['hi'], 2)})"


def join(items):
    items = list(items)
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def words(n):
    w = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen"]
    return w[n] if 0 <= n < len(w) else str(n)


def nm(m):
    return NAME.get(m, m)


def acc_cell(g, key):
    if key not in g:
        return "–"
    m, s = A(g, key)
    return f"{m:.1f} ± {s:.1f}"


def r90(m):
    key = (m, REF) if (m, REF) in main and "curve" in next(iter(main[(m, REF)].values())) else (m, "curve")
    cur = np.mean([r["curve"] for r in main[key].values()], 0)
    k = np.where(cur >= 0.9)[0]
    return int(k[0]) + 1 if len(k) else None


# ================================================================ Section 4
def results(B):
    P, H, CAP, FIG, TAB = B.para, B.heading, B.caption, B.figure, B.table
    M = main
    H("4. Results", 1)
    # ---------------------------------------------------------------- 4.1
    H("4.1 Policy comparison on the digits task", 2)
    work = ["M2", "M3", "M2-EIG", "OPT", "RCI", "C22", "Z10", "Y14", "Y4"]
    plain = sorted(["M2", "M3", "M2-EIG", "M1-BF", "OPT", "RCI", "C22", "Z10", "Y14", "Y4"], key=lambda m: -av(m))
    assert plain[0] == "C22", plain
    uvars = ["OPT-U", "RCI-U", "Z10-U", "Y14-U"]
    best_u = max(uvars, key=av)
    P(f"Table 3 and Figure 2 compare all policies. At the reference SNR of −35 dB the noiseless reference reached {IDEAL:.1f}%. Without "
      f"unit-gain normalisation, C12 was the most accurate policy ({a1('C22')}), followed by RCI ({a1('RCI')}), M3 ({a1('M3')}), M2+EIG "
      f"({a1('M2-EIG')}) and Y4 ({a1('Y4')}); M2 reached {a1('M2')}, OPT {a1('OPT')}, Z10 {a1('Z10')}, M1+BF {a1('M1-BF')} and Y14 "
      f"{a1('Y14')}. Full inversion reached only {a1('M1')}, as expected for the sanity check. The four variants with unit-gain "
      f"normalisation (suffix U, Section 4.2) reached between {min(av(m) for m in uvars):.1f}% and {max(av(m) for m in uvars):.1f}%, "
      f"so the most accurate policy overall was {nm(best_u)}. The gaps grow as the SNR falls. At −40 dB C12 reached {a1('C22', -40)}, "
      f"{nm(best_u)} {a1(best_u, -40)}, M3 {a1('M3', -40)} and M2 {a1('M2', -40)}, while at −25 dB every policy except M1, OPT, Z10 and "
      f"Y14 was within {IDEAL - min(av(m, -25) for m in ['M2', 'M3', 'M2-EIG', 'M1-BF', 'RCI', 'C22', 'Y4']):.1f} points of the "
      f"noiseless accuracy. Figure 3 shows the learning curves at −35 dB: the noiseless reference reached 90% after {r90('IDEAL')} rounds, "
      f"C12 after {r90('C22')}, OPT-U after {r90('OPT-U')}, M3 after {r90('M3')} and M2 after {r90('M2')}, while M1, Z10 and Y14 did not "
      f"reach 90% within the 100 rounds.")
    FIG(FIGS["snr"], "Figure 2.", "Test accuracy versus per-device transmit SNR on the digits task (mean of ten seeds, accuracy averaged "
        "over the last ten rounds; error bars show the standard deviation across seeds). (a) M1, M2, M3 and the two controls; (b) the "
        "policies from the literature, with M2 for reference; (c) the variants with unit-gain normalisation (Section 4.2), with C12 for "
        "reference. The dash-dotted line is the noiseless reference. Panels (b) and (c) have a narrower accuracy axis.", 16.0)
    FIG(FIGS["curves"], "Figure 3.", "Test accuracy versus communication round at the reference SNR of −35 dB (mean of ten seeds).", 12.5)
    rows = [["Policy", "Acc. −45 dB (%)", "Acc. −40 dB (%)", "Acc. −35 dB (%)", "Acc. −25 dB (%)", "MSE mean", "MSE median", "Shrink­age",
             "Part.", "Power"]]
    order = ["IDEAL", "M1", "M2", "M3", "M2-EIG", "M1-BF", "Y4", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U"]
    bold = set()
    for ri, m in enumerate(order, start=1):
        if m == "IDEAL":
            ic = acc_cell(M, ("IDEAL", REF))
            rows.append(["Ideal", ic, ic, ic, ic, "–", "–", "1.00", "1.00", "–"]); continue
        rows.append([nm(m), acc_cell(M, (m, -45)), acc_cell(M, (m, -40)), acc_cell(M, (m, REF)), acc_cell(M, (m, -25)),
                     num(F(M, (m, REF), "mse_mean"), 3 if F(M, (m, REF), "mse_mean") < 10 else 0),
                     num(tail_stats(m)["med"], 3 if tail_stats(m)["med"] < 10 else 0),
                     num(F(M, (m, REF), "proj")), num(F(M, (m, REF), "part")), num(F(M, (m, REF), "power"))])
    for ci, s in ((1, -45), (2, -40), (3, REF), (4, -25)):
        best = max(av(m, s) for m in order[1:])
        for ri, m in enumerate(order, start=1):
            if m != "IDEAL" and av(m, s) >= best - 0.05:
                bold.add((ri, ci))
    CAP("Table 3.", "Policies on the digits task with unit-power symbols (ten seeds). Accuracy is the mean of the last ten rounds ± the "
        "standard deviation across seeds (the noiseless reference does not depend on the SNR). The MSE mean and median are over all rounds "
        "and seeds; MSE, shrinkage factor ĝ^{T}ḡ/‖ḡ‖², participation ratio |S|/K and average transmit power "
        "(fraction of P_{max}) are at −35 dB. The best accuracy in each column is in bold (ties within 0.05 points).")
    TAB(rows, [940, 1060, 1060, 1060, 1060, 840, 840, 800, 640, 700], bold_cells=bold, sz=16)
    # ---------------------------------------------------------------- 4.2
    H("4.2 Lower aggregation MSE, shrinkage and the learning rate", 2)
    low = ["OPT", "Z10", "Y14"]
    for m in low:
        assert F(M, (m, REF), "mse_mean") < F(M, ("M2", REF), "mse_mean") / 5 and av(m) < av("M2")
    ratio = [F(M, ("M2", REF), "mse_mean") / F(M, (m, REF), "mse_mean") for m in low]
    P(f"Figure 4 shows the aggregation MSE against SNR as the mean and the median over rounds. OPT, Z10 and Y14 had the lowest MSE of "
      f"all policies at −35 dB, between {min(ratio):.0f} and {max(ratio):.0f} times lower than M2, yet they were less accurate than M2 "
      f"(Figure 5a). The shrinkage factor explains the gap (Figure 5b). OPT scaled the "
      f"aggregate down to {F(M, ('OPT', REF), 'proj'):.2f} of the true gradient on average, Z10 to {F(M, ('Z10', REF), 'proj'):.2f} and "
      f"Y14 to {F(M, ('Y14', REF), 'proj'):.2f}, against {F(M, ('M2', REF), 'proj'):.2f} for M2. A policy that minimises the MSE under "
      f"noise pulls the estimate towards zero, because a short estimate has less noise; for training this acts like a smaller learning "
      f"rate, so these policies converge more slowly and remain below the noiseless accuracy after 100 rounds. RCI sits in "
      f"between, with a shrinkage factor of {F(M, ('RCI', REF), 'proj'):.2f}.")
    FIG(FIGS["mse"], "Figure 4.", "Aggregation MSE per round versus per-device transmit SNR on the digits task (logarithmic scale): "
        "mean over rounds (solid lines, filled markers) and median over rounds (dotted lines, open markers), each averaged over ten "
        "seeds.", 13.0)
    P(f"To test this explanation we ran each of these policies with unit-gain normalisation: the server divides Re{{y}} by the sum of the "
      f"received amplitudes Σ_{{k∈S}}|ĉ_{{k}}b_{{k}}| instead of by |S|√η, which keeps the power control unchanged and only rescales the "
      f"estimate. The sum can be measured with one pilot symbol that all devices send with their precoders; here it is assumed known, "
      f"and Section 4.8 tests a noisy pilot. "
      f"The shrinkage factor of the U variants returned to {min(F(M, (m, REF), 'proj') for m in uvars):.2f}–"
      f"{max(F(M, (m, REF), 'proj') for m in uvars):.2f}, their MSE rose but stayed below that of M2, and their accuracy rose by "
      f"{av('OPT-U') - av('OPT'):.1f} points for OPT, {av('Z10-U') - av('Z10'):.1f} for Z10, {av('Y14-U') - av('Y14'):.1f} for Y14 and "
      f"{av('RCI-U') - av('RCI'):.1f} for RCI (Table 4). OPT-U, Z10-U and Y14-U then matched or beat C12 at −40 and −35 dB, and at −45 dB "
      f"they reached {min(av(m, -45) for m in ['OPT-U', 'Z10-U', 'Y14-U']):.1f}–{max(av(m, -45) for m in ['OPT-U', 'Z10-U', 'Y14-U']):.1f}% "
      f"against {a1('C22', -45)} for C12 and {a1('M2', -45)} for M2. The power control of the MSE-optimised policies is therefore good; "
      f"their normalisation is what costs accuracy at a fixed learning rate.")
    FIG(FIGS["shrink"], "Figure 5.", "Accuracy at −35 dB against (a) the mean aggregation MSE and (b) the shrinkage factor "
        "ĝ^{T}ḡ/‖ḡ‖² (digits task, mean of ten seeds). Open markers are the variants with unit-gain normalisation.", 16.0)
    # ---------------------------------------------------------------- 4.4
    pairs = [("M3", "M2"), ("M3", "M2-EIG"), ("Y4", "M3"), ("C22", "M3"), ("C22", "M2"), ("RCI", "M2"), ("OPT", "M2"),
             ("Z10", "M2"), ("Y14", "M2"), ("OPT-U", "OPT"), ("Z10-U", "Z10"), ("Y14-U", "Y14"), ("OPT-U", "C22"),
             ("Y14-U", "C22"), ("M2", "M1")]
    rows = [["Comparison", "−40 dB", "−35 dB", "−25 dB"]]
    res = {}
    for a, b in pairs:
        for s in (-40, REF, -25):
            res[(a, b, s)] = CI(M, (a, s), M, (b, s))
    keys = list(res)
    for k_, rj in zip(keys, T.holm([res[k_]["p"] for k_ in keys])):
        res[k_]["holm"] = rj
    for a, b in pairs:
        rows.append([f"{nm(a)} − {nm(b)}"] + [fmt_ci(res[(a, b, s)]) + ("†" if res[(a, b, s)]["holm"] else "") for s in (-40, REF, -25)])
    N["ci"] = res
    n_star = sum(not c["within_noise"] for c in res.values()); n_holm = sum(c["holm"] for c in res.values())
    lost = [k_ for k_, c in res.items() if not c["within_noise"] and not c["holm"]]
    assert all(c["holm"] <= (not c["within_noise"]) for c in res.values())
    wn35 = [f"{nm(a)} − {nm(b)}" for a, b in pairs if res[(a, b, REF)]["within_noise"]]
    sds = [A(M, (m, REF))[1] for m in ["M2", "M3", "M2-EIG", "Y4", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14-U"]]
    wn25 = [f"{nm(a)} − {nm(b)}" for a, b in pairs if res[(a, b, -25)]["within_noise"]]
    P(f"Table 4 gives the paired differences behind the main claims, with 95% confidence intervals over the ten seeds; an asterisk marks "
      f"an interval that excludes zero. At −35 dB {words(len(pairs) - len(wn35))} of the {words(len(pairs))} differences are outside "
      f"the noise; the "
      f"ones within noise are {join(wn35)}. At −25 dB, where all working policies are close to the noiseless accuracy, "
      f"{words(len(wn25))} differences are within noise ({join(wn25)}), so claims about the working policies at that SNR should not "
      f"be read into gaps of a few tenths of a point. The seed standard deviations of these policies were {min(sds):.1f} to "
      f"{max(sds):.1f} points, but because every policy sees the same data and channels, the paired intervals are much narrower. "
      f"The intervals are per comparison. Table 4 holds {words(len(res))} of them, so the {n_star} that exclude zero could include a few "
      f"false positives; with the Holm correction for all {len(res)} comparisons at a family-wise level of 5%, {n_holm} of the "
      f"{n_star} remain significant (marked †)" + (f", and the ones that drop out are {join(f'{nm(a)} − {nm(b)} at {neg(s)} dB' for a, b, s in lost)}." if lost else ".")
      + " Y4 − M3 at −35 dB, whose interval ends at zero, is not significant under either rule.")
    assert res[("Y4", "M3", REF)]["within_noise"] and not res[("Y4", "M3", REF)]["holm"]
    CAP("Table 4.", "Paired differences in accuracy (percentage points, policy A − policy B) on the digits task: mean over ten seeds "
        "[95% t-interval, per comparison]. An asterisk marks intervals that exclude zero; a dagger marks differences that remain "
        "significant after the Holm correction for all 45 comparisons (family-wise level 5%).")
    TAB(rows, [2050, 2350, 2350, 2350], center_from=1, sz=16)
    lr_section(B)
    # ---------------------------------------------------------------- 4.3
    H("4.3 Where the gain of M3 comes from", 2)
    c_eig = CI(M, ("M3", REF), M, ("M2-EIG", REF)); c_m3m2 = CI(M, ("M3", REF), M, ("M2", REF))
    c_eig40 = CI(M, ("M3", -40), M, ("M2-EIG", -40))
    c_y4 = CI(M, ("Y4", REF), M, ("M3", REF))
    c8 = CI(n8, ("M3", REF), n8, ("M2-EIG", REF))
    assert c_eig["within_noise"] and not c_m3m2["within_noise"]
    c_m3m2_40 = CI(M, ("M3", -40), M, ("M2", -40))
    assert not c_eig40["within_noise"] and c_eig40["diff"] < 0.25 * c_m3m2_40["diff"]
    P(f"With four antennas, M2+EIG, which keeps the eigenvector combiner and skips the 40 refinement steps, matched M3 at −35 dB: the "
      f"paired difference M3 − M2+EIG was {ci_txt(c_eig)}. At −40 dB the refinement added {ci_txt(c_eig40)}, a real but small part of "
      f"the {c_m3m2_40['diff']:.2f}-point gain of M3 over M2 at that SNR. The gain of M3 over M2, {ci_txt(c_m3m2)} at −35 dB, is "
      f"therefore mostly array gain from the extra antennas; the joint design adds little at N = 4. M1+BF, which uses the combiner without truncation, "
      f"reached only {a1('M1-BF')}, because with full participation the weakest effective gain still sets η. Y4, the device selection of "
      f"[4], reached {a1('Y4')}, {-c_y4['diff']:.2f} points below M3 (95% CI {sgn(c_y4['lo'], 2)} to {sgn(c_y4['hi'], 2)}). At its "
      f"tuned threshold of {N['thr']['Y4']} it kept {100 * F(M, ('Y4', REF), 'part'):.0f}% of the devices per round, against "
      f"{100 * F(M, ('M3', REF), 'part'):.0f}% for M3 at 0.7. On the tuning rounds Y4 served more devices than the combiner of M3 at a "
      f"threshold of 0.9 ({100 * TP['Y4'][0.9]:.0f}% against {100 * TP['M3'][0.9]:.0f}%) but fewer at 0.7 "
      f"({100 * TP['Y4'][0.7]:.0f}% against {100 * TP['M3'][0.7]:.0f}%): with many devices in the set, the DC algorithm sometimes stops "
      f"at a local optimum and certifies a much smaller set than the eigenvector combiner can serve.")
    P(f"With eight antennas (Table 6; the threshold retuned on the tuning rounds was again {N['thr']['N8']}) the refinement starts to "
      f"matter. M3 kept "
      f"{100 * F(n8, ('M3', REF), 'part'):.0f}% of the devices against {100 * F(n8, ('M2-EIG', REF), 'part'):.0f}% for M2+EIG, and "
      f"M3 − M2+EIG was {ci_txt(c8)} at −35 dB and {av('M3', -40, n8) - av('M2-EIG', -40, n8):.1f} points at −40 dB. M1+BF gained most "
      f"from the extra antennas, from {a1('M1-BF')} to {a1('M1-BF', REF, n8)} at −35 dB, because the weakest effective gain grows with N.")
    rows = [["Policy", "N = 4: acc. −40 dB (%)", "N = 4: acc. −35 dB (%)", "N = 4: part.", "N = 8: acc. −40 dB (%)",
             "N = 8: acc. −35 dB (%)", "N = 8: part."]]
    for m in ["M1-BF", "M2-EIG", "M3", "Y4"]:
        rows.append([nm(m), acc_cell(M, (m, -40)), acc_cell(M, (m, REF)), num(F(M, (m, REF), "part")),
                     acc_cell(n8, (m, -40)), acc_cell(n8, (m, REF)), num(F(n8, (m, REF), "part")) if (m, REF) in n8 else "–"])
    rows.append(["M2 (N = 1)", acc_cell(M, ("M2", -40)), acc_cell(M, ("M2", REF)), num(F(M, ("M2", REF), "part")), "–", "–", "–"])
    CAP("Table 6.", "Multi-antenna policies with four and eight receive antennas (digits task, ten seeds, perfect CSI).")
    TAB(rows, [1500, 1300, 1300, 950, 1300, 1300, 950], sz=16)

    # ---------------------------------------------------------------- 4.4
    results_b(B)


def results_b(B):
    P, H, CAP, FIG, TAB = B.para, B.heading, B.caption, B.figure, B.table
    M = main
    H("4.4 Full channel inversion as a sanity check", 2)
    t1 = tail_stats("M1"); rk = seed_rank("M1", REF); t2 = tail_stats("M2")
    sd1 = A(M, ("M1", REF))[1]
    P(f"M1 reached {a1('M1')} at −35 dB with a seed standard deviation of {sd1:.1f} points, and it needed about 0 dB to come within one "
      f"point of the noiseless accuracy ({a1('M1', 0)} at 0 dB). Its per-round MSE is heavy-tailed (Table 7, Figure 6): the median "
      f"over all rounds and seeds was {t1['med']:.0f}, the mean {t1['mean']:.0f} and the 99th percentile {t1['p99']:.0f}, and the 1% of "
      f"rounds with the largest error made up {100 * t1['top1']:.0f}% of the mean. For M2 the same share was {100 * t2['top1']:.0f}%. "
      f"The tail comes from η, which Eq. (6) sets from the weakest device. For M1 its median was {t1['eta_med']:.4f} and its 1st "
      f"percentile {t1['eta_p1']:.5f}, {t1['eta_med'] / t1['eta_p1']:.0f} times smaller, whereas truncation keeps η of M2 above "
      f"h_{{th}}² = {N['thr']['M2'] ** 2:.2f}. The mean MSE of a run therefore depends on how many deep fades it meets and when. Across "
      f"the ten seeds the accuracy of M1 followed the mean MSE of the run (Spearman ρ = {num(rk['mean'])}) more closely than its "
      f"median (ρ = {num(rk['med'])}), which is where the large seed variance comes from.")
    P(f"The same tail explains Figure 7, which plots accuracy against the realised receive SNR instead of the nominal SNR. At a nominal "
      f"−35 dB the realised SNR of M1 was {F(M, ('M1', REF), 'rx'):.1f} dB, {REF - F(M, ('M1', REF), 'rx'):.1f} dB below nominal, "
      f"because every device scales down to the weakest one; M1 used only {100 * F(M, ('M1', REF), 'power'):.0f}% of the power budget. "
      f"But the curves do not collapse onto one line. The realised SNR is a mean over devices and rounds and follows the typical round, "
      f"while M1 loses its accuracy in the rare rounds where η collapses, so at equal realised SNR M1 stays below the other policies. "
      f"At a nominal −20 dB, M1 had a realised SNR of {F(M, ('M1', -20), 'rx'):.1f} dB and reached {a1('M1', -20)}, while M3 at a "
      f"nominal −40 dB had a lower realised SNR, {F(M, ('M3', -40), 'rx'):.1f} dB, and still reached {a1('M3', -40)}.")
    rows = [["Policy", "MSE mean", "MSE median", "MSE 90th pct.", "MSE 99th pct.", "Top 1% share", "η median", "η 1st pct."]]
    for m in ["M1", "M1-BF", "M2", "M3", "C22"]:
        t = tail_stats(m)
        f = lambda x: num(x, 3) if x < 10 else (num(x, 1) if x < 100 else num(x, 0))
        rows.append([nm(m), f(t["mean"]), f(t["med"]), f(t["p90"]), f(t["p99"]), f"{100 * t['top1']:.0f}%",
                     f"{t['eta_med']:.4f}", f"{t['eta_p1']:.5f}"])
    CAP("Table 7.", "Distribution of the per-round aggregation MSE and of the alignment factor η at −35 dB (digits task, all 100 rounds "
        "of all ten seeds). 'Top 1% share' is the fraction of the mean MSE contributed by the 1% of rounds with the largest error. For "
        "C12 η is the fixed target τ².")
    TAB(rows, [1150, 1100, 1100, 1150, 1150, 1100, 1100, 1150], sz=16)
    FIG(FIGS["tail"], "Figure 6.", "Fraction of rounds whose aggregation MSE exceeds a given value at −35 dB (all rounds of ten seeds, "
        "logarithmic axes).", 12.5)
    FIG(FIGS["rxsnr"], "Figure 7.", "Test accuracy against the realised receive SNR (digits task, perfect CSI, mean of ten seeds). Each "
        "marker is one nominal SNR from −45 to 20 dB.", 12.5)
    # ---------------------------------------------------------------- 4.5
    H("4.5 Robustness: CSI error, time-correlated fading and label skew", 2)
    d = lambda m, e: av(m) - A(csi, (m, e))[0]
    P(f"Figure 8 shows accuracy as the CSI error variance ε grows from 0 to 0.4 at −35 dB, and Table 8 collects the robustness checks. "
      f"With ε = 0.4, M2 lost {d('M2', 0.4):.1f} points and M3 {d('M3', 0.4):.1f}, so the extra antennas also make the policy less "
      f"sensitive to estimation error; M1+BF, which keeps the weakest devices, lost {d('M1-BF', 0.4):.1f}. The policies that let every "
      f"device transmit with a capped amplitude were the least affected: C12 lost {d('C22', 0.4):.1f}, OPT-U {d('OPT-U', 0.4):.1f} and "
      f"Y14-U {d('Y14-U', 0.4):.1f} points, and Y4 lost {d('Y4', 0.4):.1f} points.")
    jd = lambda m, fd: av(m) - A(jk, (m, fd))[0]
    worst_j = max(["M2", "M3", "M2-EIG", "OPT", "OPT-U", "RCI", "C22", "Z10", "Z10-U", "Y14-U"], key=lambda m: abs(jd(m, 0.01)))
    P(f"Time-correlated fading changed little for most policies. With f_{{D}}T = 0.01, where the channels hardly change over the 100 "
      f"rounds and the same devices stay weak, the working policies other than Y14 and M1+BF moved by at most "
      f"{abs(jd(worst_j, 0.01)):.1f} points, and M1+BF lost {jd('M1-BF', 0.01):.1f}. "
      f"Y14 lost {jd('Y14', 0.01):.1f} points at f_{{D}}T = 0.01 and {jd('Y14', 0.05):.1f} at f_{{D}}T = 0.05, and M1+BF lost "
      f"{jd('M1-BF', 0.05):.1f} points at f_{{D}}T = 0.05. Y14 down-weights a weak device in every round, and with slow fading the same "
      f"devices are down-weighted round after round, so their data are under-represented in the model. With unit-gain normalisation Y14 was again unaffected ({a1('Y14-U', 0.01, jk)} at "
      f"f_{{D}}T = 0.01).")
    ad = lambda m: av(m) - A(al, (m, 0.1))[0]
    P(f"Label skew matters more. With α = 0.1 the median device has {words(int(SKEW[0.1]))} classes that each make up at least 10% of "
      f"its data, against {words(int(SKEW[0.5]))} with α = 0.5. The noiseless reference was unchanged "
      f"({A(al, ('IDEAL', 0.1))[0]:.1f}%), but M2 lost {ad('M2'):.1f} points and M3 {ad('M3'):.1f}, whereas C12 lost {ad('C22'):.1f}, "
      f"OPT-U {ad('OPT-U'):.1f} and Y14-U {ad('Y14-U'):.1f}. Truncation drops the weakest devices, and when each device holds different "
      f"classes, the classes of the dropped devices are missing from that round's update; the policies that keep every device in the sum "
      f"avoid this. With nearly IID data (α = 100) the differences shrink: M2 reached {a1('M2', 100.0, al)}, M3 {a1('M3', 100.0, al)} "
      f"and C12 {a1('C22', 100.0, al)}.")
    FIG(FIGS["csi"], "Figure 8.", "Test accuracy versus CSI error variance ε at −35 dB (digits task, mean of ten seeds).", 12.5)
    rows = [["Policy", "Reference", "ε = 0.4", "f_{D}T = 0.01", "f_{D}T = 0.05", "α = 0.1", "α = 100"]]
    for m in ["IDEAL", "M1", "M2", "M3", "M2-EIG", "M1-BF", "Y4", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U"]:
        cell = lambda g, k: f"{A(g, k)[0]:.1f}" if k in g else "–"
        rows.append(["Ideal" if m == "IDEAL" else nm(m), cell(M, (m, REF)), cell(csi, (m, 0.4)) if m != "IDEAL" else "–",
                     cell(jk, (m, 0.01)), cell(jk, (m, 0.05)), cell(al, (m, 0.1)), cell(al, (m, 100.0))])
    CAP("Table 8.", "Accuracy (%) at −35 dB under channel-estimation error, time-correlated Gauss–Markov fading with Jakes correlation "
        "and different label skew (digits task, ten seeds; the reference column is i.i.d. fading, perfect CSI and α = 0.5). Y4 was "
        "run only with estimation error.")
    TAB(rows, [1400, 1250, 1200, 1300, 1300, 1150, 1150], sz=16)
    # ---------------------------------------------------------------- 4.6
    H("4.6 The truncation threshold and the threshold rule", 2)
    tr = {t: (F(thr, (t,), "mse_mean"), F(thr, (t,), "noise"), F(thr, (t,), "part"), A(thr, (t,))[0]) for (t,) in thr}
    ths = sorted(tr)
    best_m = min(ths, key=lambda t: tr[t][0]); best_a = max(ths, key=lambda t: tr[t][3])
    rr = rule_rows()
    assert all(r["mse_opt"] == 0.7 for r in rr)
    P(f"Figure 9 sweeps the threshold h_{{th}} of M2 at −35 dB. Without truncation (h_{{th}} = 0, which is M1) the MSE was "
      f"{tr[0.0][0]:.0f}; a threshold of 0.1, which silenced {100 * (1 - tr[0.1][2]):.1f}% of the devices, cut it to {tr[0.1][0]:.1f}. "
      f"The MSE was lowest at h_{{th}} = {best_m} ({tr[best_m][0]:.2f}, participation {tr[best_m][2]:.2f}) and rose again to "
      f"{tr[1.2][0]:.2f} at h_{{th}} = 1.2, where only {100 * tr[1.2][2]:.0f}% of the devices transmitted. Accuracy peaked at "
      f"h_{{th}} = {best_a} ({tr[best_a][3]:.2f}%), level with h_{{th}} = 0.7 ({tr[0.7][3]:.2f}%). Almost all of the error is noise: at "
      f"h_{{th}} = 0.7 the noise term of Eq. (8) was {tr[0.7][1]:.2f} of the total {tr[0.7][0]:.2f}, so the rise at large thresholds "
      f"comes from the shrinking number of participants, not from bias.")
    P(f"The threshold rule of Section 3.5 gives h_{{th}} = {rr[0]['rule']:.2f}, {rr[1]['rule']:.2f} and {rr[2]['rule']:.2f} for K = 20, "
      f"50 and 100, and the grid value with the lowest MSE was 0.7 for all three (Table 9). The rule ignores the spread of the per-symbol "
      f"powers π_{{k}}, which is zero with unit-power symbols. In accuracy the best grid value was "
      f"{rr[0]['acc_opt']}, {rr[1]['acc_opt']} and {rr[2]['acc_opt']}, and the threshold of 0.7 was at most "
      f"{max(r['acc_best'] - r['acc07'] for r in rr):.2f} points behind. A fixed threshold of 0.7 is therefore close to the best choice "
      f"for all three sizes, and the tuning run is not needed. We checked one SNR, single-antenna truncation and Rayleigh fading only.")
    FIG(FIGS["threshold"], "Figure 9.", "Effect of the truncation threshold h_{th} of M2 at −35 dB (digits task, ten seeds). Left axis "
        "(logarithmic): total aggregation MSE and its noise-only component; right axis: participation ratio |S|/K.", 12.5)
    rows = [["K", "Rule h_{th}", "MSE-minimising h_{th} (grid)", "MSE at that value", "Accuracy-maximising h_{th} (grid)",
             "Accuracy at h_{th} = 0.7 (%)"]]
    for r in rr:
        rows.append([str(r["K"]), num(r["rule"]), str(r["mse_opt"]), num(r["mse_min"], 3), str(r["acc_opt"]), f"{r['acc07']:.2f}"])
    CAP("Table 9.", "Threshold rule against a grid search of M2 at −35 dB with perfect CSI (ten seeds). The grid was 0, 0.2, 0.3, …, 1.0, "
        "1.2 and 1.5.")
    TAB(rows, [700, 1200, 1700, 1500, 1800, 1700], sz=16)
    larger(B)
    ablation(B)


# ================================================================ per-policy learning rate (minor corrections, item 1)
def drift(m, snr=REF):
    """mean shrinkage factor of policy m relative to its unit-gain variant in rounds 1-10 and 61-100 (tuned rates)"""
    pc = lambda lab: np.mean([r["proj_curve"] for r in lrr[(lab, snr)].values()], 0)
    a_, b_ = pc(m), pc(m + "-U")
    return float(a_[:10].mean() / b_[:10].mean()), float(a_[60:].mean() / b_[60:].mean())


def lr_section(B):
    P, CAP, TAB = B.para, B.caption, B.table
    L, g = N["lr"], lrr
    snrs = [-45, -40, -35, -30, -25]
    la = lambda m, s=REF: A(g, (m, s))[0]
    lci = lambda a_, b_, s=REF: CI(g, (a_, s), g, (b_, s))
    mso = ["OPT", "Z10", "Y14"]
    for m in mso:
        assert L[m] >= 2 * L[m + "-U"]
        assert min(N["lr_grid"]) < L[m] < max(N["lr_grid"])
    c_opt = {s: lci("OPT", "OPT-U", s) for s in snrs}
    assert all(c["within_noise"] or abs(c["diff"]) < 0.6 for c in c_opt.values())
    c_y35, c_y45 = lci("Y14", "Y14-U"), lci("Y14", "Y14-U", -45)
    c_z35, c_z45 = lci("Z10", "Z10-U"), lci("Z10", "Z10-U", -45)
    assert c_y35["within_noise"] and c_y45["lo"] > 0 and c_z35["hi"] < 0 and c_z45["lo"] > 0
    best45 = sorted([m for m in L if m not in ("IDEAL", "M1")], key=lambda m: -la(m, -45))
    assert best45[0] == "Y14", best45
    c45 = lci("Y14", "C22", -45)
    z0, z1 = drift("Z10"); o0, o1 = drift("OPT")
    work = ["M2", "M3", "M2-EIG", "M1-BF", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U"]
    top35 = max(la(m) for m in work)
    P(f"A shrinkage factor below one works like a smaller learning rate, and all policies so far used the same server learning rate "
      f"of 1.0, which suits the policies with little shrinkage. To separate the two effects we tuned the learning rate of every policy "
      f"on the tuning rounds at −35 dB (five seeds, rates {num(min(N['lr_grid']), 4).rstrip('0')} to {max(N['lr_grid']):.0f} in "
      f"factors of two) and ran all policies at their own rates on the reporting rounds (Table 5). The rates that the policies chose "
      f"follow their shrinkage: {L['M2']:g} for M2, {L['M3']:g} for M3, {L['C22']:g} for C12 and the U variants of OPT, Z10 and Y14, but "
      f"{L['OPT']:g} for OPT, {L['Z10']:g} for Z10 and {L['Y14']:g} for Y14. The noiseless reference also gained from a larger rate "
      f"({la('IDEAL'):.1f}% at {L['IDEAL']:g}), so the digits task is not saturated at 1.0.")
    P(f"With its own learning rate, OPT was level with OPT-U at every SNR: the paired difference OPT − OPT-U was "
      f"{ci_txt(c_opt[REF])} at −35 dB and at most {max(abs(c['diff']) for c in c_opt.values()):.1f} points elsewhere. Y14 − Y14-U was "
      f"{ci_txt(c_y35)} at −35 dB and {ci_txt(c_y45)} at −45 dB. Z10 stayed behind Z10-U at −35 dB, where Z10 − Z10-U was "
      f"{ci_txt(c_z35)}, but was ahead at −45 dB, {ci_txt(c_z45)}. One reason is that the shrinkage of Z10 changes during training: relative to Z10-U it fell from "
      f"{z0:.2f} in the first ten rounds to {z1:.2f} in the last forty, whereas that of OPT stayed at {o0:.2f} to {o1:.2f}, so no "
      f"single rate fits Z10 in every round. At −45 dB the tuned MSE-optimised policies were the most accurate of all: Y14 reached "
      f"{la('Y14', -45):.1f}%, Z10 {la('Z10', -45):.1f}% and OPT {la('OPT', -45):.1f}%, against {la('C22', -45):.1f}% for C12 "
      f"and {la('M2', -45):.1f}% for M2; Y14 − C12 was {ci_txt(c45)}. From −35 dB upward the working policies were within "
      f"{top35 - min(la(m) for m in work if m != 'M1-BF'):.1f} points of each other, apart from M1+BF.")
    P("This narrows the claim of the comparison at a fixed learning rate. MSE-optimised power control serves learning well once the "
      "bias of its estimate towards zero is removed, either in the normalisation (the U variants, which need "
      "the amplitude sum) or in the step size (which needs a learning rate tuned for the policy). A benchmark that uses one learning "
      "rate for all policies, as most do, ranks the MSE-optimised policies too low. The rates were tuned at −35 dB only, and the "
      "best rate of a shrunk policy depends on the SNR through its shrinkage: at −25 dB Y14 at its tuned rate was "
      f"{la('Y14-U', -25) - la('Y14', -25):.1f} points behind Y14-U.")
    assert la("Y14-U", -25) > la("Y14", -25)
    rows = [["Policy", "Learning rate"] + [f"{neg(s)} dB" for s in snrs] + ["Shrink­age −35 dB"]]
    order = ["IDEAL", "M2", "M3", "M2-EIG", "M1-BF", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U"]
    bold = set()
    for ri, m in enumerate(order, start=1):
        if m == "IDEAL":
            ic = acc_cell(g, ("IDEAL", REF))
            rows.append(["Ideal", f"{L['IDEAL']:g}", ic, ic, ic, ic, ic, "1.00"]); continue
        rows.append([nm(m), f"{L[m]:g}"] + [acc_cell(g, (m, s)) for s in snrs] + [num(F(g, (m, REF), "proj"))])
    for ci_, s in enumerate(snrs, start=2):
        best = max(la(m, s) for m in order[1:])
        for ri, m in enumerate(order, start=1):
            if m != "IDEAL" and la(m, s) >= best - 0.05:
                bold.add((ri, ci_))
    CAP("Table 5.", f"Every policy at its own server learning rate, tuned on the tuning rounds at −35 dB (digits task, ten seeds, "
        f"reporting rounds; accuracy as in Table 3). M1, whose best rate was the smallest one tried ({L['M1']:g}), reached "
        f"{la('M1'):.1f}% at −35 dB and is not listed. The best accuracy in each column is in bold (ties within 0.05 points).")
    TAB(rows, [1100, 950, 1150, 1150, 1150, 1150, 1150, 1100], bold_cells=bold, sz=16)


# ================================================================ larger tasks
def load_big(fname, seeds=None):
    """{(method, snr): {seed: accuracy %}} plus other fields; accuracy = mean of the last three evaluations"""
    out = collections.defaultdict(dict)
    if not os.path.exists(V(fname)):
        return out
    for line in open(V(fname)):
        r = json.loads(line)
        if r["seed"] == 9 or (seeds is not None and r["seed"] not in seeds):
            continue                                    # seed 9 is the tuning seed
        if fname.startswith("cifar") and abs(r.get("lr", 0.2) - 0.2) > 1e-9:
            continue
        r["acc3"] = pct(float(np.mean(r["acc"][-3:])))
        out[(r["method"], float(r["snr"]))][r["seed"]] = r
    return out


MN = load_big("mnist_v3.jsonl")
CF = load_big("cifar_v3.jsonl")


def bacc(g, key):
    v = [r["acc3"] for r in g[key].values()]
    return T.mean_sd(v) if len(v) else (float("nan"), float("nan"))


def bci(g, a, b, s):
    common = set(g[(a, s)]) & set(g[(b, s)])
    if len(common) < 2:
        return dict(n=len(common), diff=float("nan"), lo=float("nan"), hi=float("nan"), within_noise=True)
    return T.paired_ci({k: r["acc3"] for k, r in g[(a, s)].items()}, {k: r["acc3"] for k, r in g[(b, s)].items()})


def bfield(g, key, f):
    return float(np.mean([r[f if f in next(iter(g[key].values())) else "mse"] for r in g[key].values()]))


def bcell(g, key):
    if key not in g or not g[key]:
        return "–"
    m, s = bacc(g, key)
    return f"{m:.1f} ± {s:.1f}"


def ablation(B):
    P, H, CAP, TAB = B.para, B.heading, B.caption, B.table
    M = main
    O = old_fixed()
    H("4.8 Ablations: symbol scaling, peak power and the amplitude pilot", 2)
    pol = ["M1", "M2", "M3", "M2-EIG", "M1-BF", "OPT", "RCI", "C22"]
    rank_f = sorted(pol, key=lambda m: -O[m][0]); rank_u = sorted(pol, key=lambda m: -av(m))
    assert rank_f[0] == rank_u[0] == "C22" and rank_f[-1] == rank_u[-1] == "M1"
    worse = [m for m in pol if O[m][0] < av(m)]
    P(f"Table 12 repeats the comparison with the fixed scaling s_{{k}} = g_{{k}}/ρ, ρ = 0.1, of our earlier version. There the per-symbol "
      f"power π_{{k}} is below 1 and differs between devices and rounds, and the thresholds and constants were tuned for that convention "
      f"at −25 dB (h_{{th}} = 0.5 for M2 and 0.7 for M3, λ = 0.05 for RCI, κK = 10 and τ = 2 for C12). At the same nominal SNR "
      f"{'all ' + words(len(pol)) if len(worse) == len(pol) else words(len(worse)) + ' of the ' + words(len(pol))} policies were less "
      f"accurate with fixed scaling at −35 dB, because the symbols use less than the "
      f"full power budget; M2, for example, reached {O['M2'][0]:.1f}% against {a1('M2')}. The main conclusions hold under both "
      f"conventions: C12 was the most accurate and M1 the least accurate of these policies, M2+EIG and M3 were level, and OPT had a much "
      f"lower MSE than M2 without a higher accuracy at −25 dB ({O['OPT'][1]:.1f}% against {O['M2'][1]:.1f}%). The order of OPT and RCI "
      f"relative to M2 at −35 dB did change: with fixed scaling OPT was ahead of M2 and RCI behind it, and with unit power the reverse.")
    rows = [["Policy", "Acc. −35 dB fixed / unit (%)", "Acc. −25 dB fixed / unit (%)", "MSE −25 dB fixed / unit"]]
    for m in pol:
        rows.append([nm(m), f"{O[m][0]:.1f} / {av(m):.1f}", f"{O[m][1]:.1f} / {av(m, -25):.1f}",
                     f"{num(O[m][2], 3 if O[m][2] < 10 else 1)} / {num(F(M, (m, -25), 'mse_mean'), 3 if F(M, (m, -25), 'mse_mean') < 10 else 1)}"])
    CAP("Table 12.", "Fixed symbol scaling (ρ = 0.1, earlier version) against unit-power symbols on the digits task, perfect CSI, ten "
        "seeds. Accuracy is the mean of the last ten rounds.")
    TAB(rows, [1500, 2500, 2500, 2500], sz=16)
    ablation_b(B)


def ablation_b(B):
    P, CAP, TAB = B.para, B.caption, B.table
    M, g = main, rmx
    pol = ["M2", "M3", "M2-EIG", "C22", "RCI", "OPT", "OPT-U", "RCI-U", "Y14-U"]
    d35 = {m: CI(g, (m, REF), M, (m, REF)) for m in pol}
    worse = [m for m in pol if d35[m]["hi"] < 0]
    big = sorted(pol, key=lambda m: d35[m]["diff"])
    pw = lambda gg, m: F(gg, (m, REF), "power")
    floor_mean, floor_max = F(M, ("M2", 20), "mse_mean"), F(g, ("M2", 20), "mse_mean")
    assert floor_max > 0.8 * floor_mean and F(g, ("M2", 20), "proj") > 0.98
    P(f"The unit-power convention multiplies the received sum by the mean of ρ_{{k}}, which gives every participating gradient the same "
      f"norm and caps the shrinkage factor near {N['shrink_hi']:.2f}. The common scaling s_{{k}} = g_{{k}}/ρ_{{max}}, with ρ_{{max}} the "
      f"largest ρ_{{k}} of the round sent to the devices as side information, removes this bias: the shrinkage factor of M2 rose from "
      f"{F(M, ('M2', REF), 'proj'):.2f} to {F(g, ('M2', REF), 'proj'):.2f} at −35 dB. It costs power, because only the device with the "
      f"largest gradient uses its full budget, and the average transmit power of M2 fell from {pw(M, 'M2'):.2f} to {pw(g, 'M2'):.2f} of "
      f"P_{{max}}. At −35 dB {words(len(worse))} of the {words(len(pol))} policies in Table 13 lost accuracy with the common scaling, "
      f"M2 by {-d35['M2']['diff']:.1f} points and C12 by {-d35['C22']['diff']:.1f}; at −25 dB the two conventions were within "
      f"{max(abs(A(g, (m, -25))[0] - A(M, (m, -25))[0]) for m in pol):.1f} points. Z10 normalises by the mean squared gradient norm "
      f"(Section 3.4) and is not affected. The M2 MSE floor at 20 dB changed only from {floor_mean:.4f} to {floor_max:.4f}: most of it "
      f"is the bias of truncation, because {100 * (1 - F(M, ('M2', 20), 'part')):.0f}% of the devices stay silent.")
    rows = [["Policy", "Acc. −40 dB mean / max (%)", "Acc. −35 dB mean / max (%)", "Acc. −25 dB mean / max (%)", "Shrinkage −35 dB",
             "Power −35 dB"]]
    for m in pol:
        rows.append([nm(m)] + [f"{A(M, (m, s))[0]:.1f} / {A(g, (m, s))[0]:.1f}" for s in (-40, REF, -25)]
                    + [f"{F(M, (m, REF), 'proj'):.2f} / {F(g, (m, REF), 'proj'):.2f}", f"{pw(M, m):.2f} / {pw(g, m):.2f}"])
    CAP("Table 13.", "Mean-ρ rescaling (s_{k} = g_{k}/ρ_{k}, used throughout) against the common scaling s_{k} = g_{k}/ρ_{max} on the "
        "digits task (ten seeds, perfect CSI, fixed learning rate 1.0, thresholds and constants as in Table 2). Each cell gives "
        "mean ρ / ρ_{max}.")
    TAB(rows, [1150, 1650, 1650, 1650, 1450, 1450], sz=16)
    # peak-limited Y14 and the noisy amplitude pilot
    yc = lambda m, s: CI(y14c, (m, s), M, (m, s))
    pl = {m: {s: CI(plt_, (m, s, 0.0), M, (m, s)) for s in (-45, -40, REF, -30, -25)} for m in ("OPT-U", "RCI-U", "Z10-U", "Y14-U")}
    ple = {m: CI(plt_, (m, REF, 0.2), csiu, (m, 0.2)) for m in pl}
    assert all(pl[m][-45]["diff"] < pl[m][REF]["diff"] for m in pl)
    P(f"Y14 has no per-round peak limit and reached {N['y14_peak']:.1f} P_{{max}} at −35 dB. With the same peak limit as the other "
      f"policies (|b_{{k}}|²π_{{k}} ≤ P_{{max}} in every round), Y14 lost {-yc('Y14', REF)['diff']:.1f} points at −35 dB and "
      f"{-yc('Y14', -45)['diff']:.1f} at −45 dB, and Y14-U lost {-yc('Y14-U', REF)['diff']:.2f} and {-yc('Y14-U', -45)['diff']:.1f} "
      f"points (Table 14). The peak budget helps Y14 most at low SNR; with the limit, Y14-U stayed within "
      f"{-yc('Y14-U', REF)['diff']:.1f} points of its unlimited version from −35 dB upward.")
    P(f"The U variants so far used the exact amplitude sum Σ_{{k∈S}}|ĉ_{{k}}b_{{k}}|, computed from the channel estimates. A server that "
      f"measures it with one pilot symbol, sent by all devices with their precoders, receives |Σ_{{k∈S}}c_{{k}}b_{{k}} + n| through the "
      f"true channels and with receiver noise. With such a pilot the U variants lost {min(-pl[m][-30]['diff'] for m in pl):.1f} to "
      f"{max(-pl[m][-30]['diff'] for m in pl):.1f} points at −30 dB, {min(-pl[m][REF]['diff'] for m in pl):.1f} to "
      f"{max(-pl[m][REF]['diff'] for m in pl):.1f} at −35 dB and {min(-pl[m][-45]['diff'] for m in pl):.1f} to "
      f"{max(-pl[m][-45]['diff'] for m in pl):.1f} at −45 dB, and with CSI error ε = 0.2 they lost "
      f"{min(-ple[m]['diff'] for m in ple):.1f} to {max(-ple[m]['diff'] for m in ple):.1f} points against the exact sum. One pilot "
      f"symbol therefore costs little from −30 dB upward; at lower SNR the pilot needs more energy, for example several symbols "
      f"averaged, before unit-gain normalisation pays off.")
    assert max(-pl[m][-30]['diff'] for m in pl) < 0.8
    rows = [["Policy", "−45 dB", "−40 dB", "−35 dB", "−35 dB, ε = 0.2"]]
    for m in ("OPT-U", "RCI-U", "Z10-U", "Y14-U"):
        rows.append([f"{nm(m)}: exact / pilot"] + [f"{A(M, (m, s))[0]:.1f} / {A(plt_, (m, s, 0.0))[0]:.1f}" for s in (-45, -40, REF)]
                    + [f"{A(csiu, (m, 0.2))[0]:.1f} / {A(plt_, (m, REF, 0.2))[0]:.1f}"])
    for m in ("Y14", "Y14-U"):
        rows.append([f"{nm(m)}: no peak limit / peak limit"] + [f"{A(M, (m, s))[0]:.1f} / {A(y14c, (m, s))[0]:.1f}" for s in (-45, -40, REF)]
                    + ["–"])
    CAP("Table 14.", "Accuracy (%) of the U variants with the exact amplitude sum against one noisy pilot symbol, and of Y14 without "
        "and with the per-round peak limit (digits task, ten seeds, fixed learning rate 1.0).")
    TAB(rows, [2900, 1500, 1500, 1500, 1700], sz=16)


# ================================================================ front-matter blocks
def _abstract():
    c_m3 = CI(main, ("M3", REF), main, ("M2", REF))
    lo_p = min(F(main, (m, REF), "proj") for m in ("OPT", "Z10", "Y14"))
    hi_p = max(F(main, (m, REF), "proj") for m in ("OPT", "Z10", "Y14"))
    u = [av(m) for m in ("OPT-U", "Z10-U", "Y14-U")]
    return (
        "In over-the-air federated learning, the power-control policy decides how fading distorts the superposed model updates. "
        "We compare such policies under Rayleigh fading on one benchmark: full channel inversion (M1), truncated inversion "
        "(M2), truncated inversion with a four-antenna receiver (M3), two controls, re-implementations of five policies from the "
        "literature and one policy inspired by a sixth. Symbols have unit power, so the SNR is the per-device transmit SNR, and differences are paired 95% confidence "
        f"intervals over ten seeds. With 50 non-IID devices training a softmax model at −35 dB, M2 reached {av('M2'):.1f}%, M3 "
        f"{av('M3'):.1f}% and the noiseless reference {IDEAL:.1f}%. Full inversion, kept as a sanity check, reached {av('M1'):.1f}%, "
        "its mean error dominated by rare deep-fade rounds. At a common learning rate the three policies with the lowest "
        f"aggregation MSE were, at −35 dB and above, among the least accurate ({av('OPT'):.1f}%, {av('Z10'):.1f}% and "
        f"{av('Y14'):.1f}% at −35 dB), because they shrink the aggregated gradient to {100 * lo_p:.0f}–{100 * hi_p:.0f}% of its length. "
        f"Dividing by the sum of the received amplitudes raised them to {min(u):.1f}–{max(u):.1f}%, and a learning rate tuned per "
        "policy worked similarly: the MSE-optimal estimate is biased towards zero, and the bias must be compensated in the "
        f"normalisation or the step size. The gain of M3 over M2, {c_m3['diff']:.2f} points, is mostly array gain. Capped inversion was more robust than truncation to strong label skew, a threshold rule "
        "that needs only the number of devices matched the best grid value, and " + BIG_SENTENCE + ".")


CONTRIB = [
    "A benchmark of power-control policies for OTA-FL in which M1, M2 and M3, two controls, re-implementations of the policies of "
    "[7], [10] and [12] and of the device selection of [4], and a policy inspired by [14] run on the same data and channels, with "
    "unit-power symbols and paired confidence intervals.",
    "Evidence, with the mechanism measured, that at a common learning rate a lower aggregation MSE does not give a higher accuracy: "
    "the MSE-optimised policies shrink the aggregated gradient. Removing the shrinkage, by unit-gain normalisation or by a learning "
    "rate tuned per policy, makes them among the most accurate: the MSE-optimal estimate is biased towards zero, and the bias "
    "can be compensated.",
    "Controls showing that the gain of the multi-antenna policy M3 over single-antenna truncation is mostly array gain with four "
    "antennas, "
    "with eight antennas and the selection algorithm of [4] as further checks.",
    "A threshold rule that needs only the number of devices, checked for K = 20, 50 and 100, and robustness checks under "
    "channel-estimation error, time-correlated fading, strong and weak label skew, MNIST with a small CNN and CIFAR-10 with a "
    "136,874-parameter CNN.",
]

TABLE1_ROW = ["This study", "Eleven policies and four unit-gain variants: M1–M3, controls, re-implemented [4], [7], [10], [12], a policy inspired by [14]",
              "Perfect and imperfect", "Test accuracy, with MSE, shrinkage and power",
              "Simulation only; small models; constants of [10] and [12] tuned or replaced; [14] not re-implemented"]

TABLE2 = [["Parameter", "Value"],
          ["Devices K / rounds per run / random seeds", "50 / 100 / 10"],
          ["Training / test samples (scikit-learn digits)", "1,347 / 450"],
          ["Model and size", "Softmax regression, d = 650"],
          ["Data heterogeneity", "Dirichlet, α = 0.5 (0.1 and 100 in Section 4.5)"],
          ["Channel", "Rayleigh, h ~ CN(0, 1); i.i.d., or Gauss–Markov with f_{D}T = 0.01 and 0.05"],
          ["Receive antennas (M3, M2+EIG, M1+BF, Y4)", "4 (8 in Section 4.3)"],
          ["Symbols / power budget P_{max} / server learning rate", "Unit power / 1 / 1.0 (per policy in Section 4.2)"],
          ["SNR sweep (reference SNR)", "−45 to 20 dB (−35 dB)"],
          ["CSI error variance ε", "0 to 0.4"],
          ["Threshold h_{th}: M2 / M3 and M2+EIG / Y4 / M3 with N = 8", None],
          ["RCI λ / C12 κK and τ", None],
          ["Channel realisations: tuning / reporting", "0–99 / 400–499"],
          ["MNIST: model / K / seeds", "CNN, 13,610 parameters / 100 / 5 at −35 dB, 3 at −45, −40 and −30 dB"],
          ["CIFAR-10: model / K / rounds / seeds", None]]
TABLE2[10][1] = f"{N['thr']['M2']} / {N['thr']['M3']} / {N['thr']['Y4']} / {N['thr']['N8']}"
TABLE2[11][1] = f"{N['thr']['RCI']} / {N['c12_kK']:.0f} and {N['thr']['C22']}"


# ================================================================ Sections 5 and 6
def discussion(B):
    P, H = B.para, B.heading
    M = main
    t1 = tail_stats("M1")
    d01 = lambda m: av(m) - A(al, (m, 0.1))[0]
    H("5. Discussion", 1)
    L = N["lr"]
    P(f"Three results stand out. The first is that the aggregation MSE, the objective of most power-control designs [7], [8], [10], "
      f"ranked the policies wrongly when every policy used the same learning rate. OPT, Z10 and Y14 had the lowest MSE and, from −35 dB "
      f"upward, some of the lowest accuracies, because under noise the estimate with the smallest MSE is a shrunk one, and a shrunk "
      f"gradient slows training in the same way as a smaller learning rate. The shrinkage factor predicted the ordering where the MSE "
      f"did not. Both remedies worked: unit-gain normalisation, which keeps the power control and only rescales the estimate, and a "
      f"learning rate tuned per policy, which for OPT, Z10 and Y14 came out {L['OPT']:g}, {L['Z10']:g} and {L['Y14']:g} times the common "
      f"rate. The contribution is therefore a smaller one than a new power-control policy: MSE-optimal estimators are biased, so a "
      f"benchmark must either remove the bias or tune the step size for each policy, and it should report the shrinkage of the aggregate "
      f"next to its MSE. This supports the argument behind [12] and [14] that power control should be judged on the learning objective.")
    loss = {m: d01(m) for m in ("M1-BF", "M2", "M3", "M2-EIG", "OPT", "OPT-U", "RCI", "RCI-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U")}
    order_ = sorted(loss, key=lambda m: -loss[m])
    assert order_[0] == "Y14" and order_[1] == "Z10", order_
    P(f"The second is that keeping every device with a capped amplitude did better than truncation. C12 and the U variants let a weak "
      f"device transmit at full power, so its data stay in every update at reduced weight, whereas M2 and M3 drop it. At −35 dB with "
      f"α = 0.5, C12 was {av('C22') - av('M2'):.1f} points ahead of M2; with α = 0.1, M2 lost {d01('M2'):.1f} points and C12 only "
      f"{d01('C22'):.1f}. This fits the bias that truncation adds when the omitted devices hold different data [3], one of the costs "
      f"of truncated inversion listed in [31]. Truncation keeps two practical advantages: each device needs only "
      f"its own channel and a threshold, and the threshold follows from K alone.")
    P(f"The third is that the multi-antenna gain of M3 is mostly array gain at N = 4. The eigenvector combiner did as well as the refined "
      f"one at −35 dB and was {CI(M, ('M3', -40), M, ('M2-EIG', -40))['diff']:.2f} points behind at −40 dB, and the device selection of "
      f"[4], solved with its DC algorithm, did not do better. With eight antennas the refinement raised the "
      f"participation from {100 * F(n8, ('M2-EIG', REF), 'part'):.0f}% to {100 * F(n8, ('M3', REF), 'part'):.0f}% and the accuracy by "
      f"{av('M3', REF, n8) - av('M2-EIG', REF, n8):.1f} points, so a joint design starts to pay as the number of antennas grows.")
    P(f"Full inversion fails in the way Eqs. (6) and (8) predict. η is set by the weakest of 50 exponentially distributed gains, whose "
      f"mean is 1/50 of the average gain, and its lower tail is much heavier still, so a few rounds with a tiny η carry most of the "
      f"error: {100 * t1['top1']:.0f}% of the mean MSE came from 1% of the rounds. A study that reports only the mean MSE of M1 hides "
      f"this, and the seed variance of M1 follows from it. Convergence analyses of learning over noisy or fading aggregation [16], "
      f"[17], [20] show that the aggregation error enters the convergence bound; our results add that a deterministic shrinkage of the "
      f"update acts through the step size and can matter more than the noise. Scheduling studies such as [22] and [24] treat the choice "
      f"of participating devices as a design variable, which is the role of the threshold here. Z10 follows [10] in treating the "
      f"gradients of different devices as identically distributed; under strong label skew that assumption fails, and Z10 lost "
      f"{d01('Z10'):.1f} points with α = 0.1, the second largest loss after Y14.")
    P("For a system designer three settings follow. If the server can measure the sum of the received amplitudes, an MSE-optimised "
      "policy with unit-gain normalisation (OPT-U, Z10-U or Y14-U) or the capped inversion C12 was among the most accurate choices on "
      "the digits task; one pilot symbol is enough for the measurement from −30 dB upward, and below that the pilot needs more energy. "
      "If the learning rate can be tuned for the deployed policy, the MSE-optimised policies need no extra measurement and were the "
      "most accurate at −45 dB. If only local channel knowledge is available, truncation with h_{th} = 0.7 from the threshold rule is a "
      "safe default, and extra receive antennas reduce its loss under estimation error.")
    P(f"The study has several limitations. All results come from simulation, with perfect synchronisation and error-free side "
      f"information: the norms ρ_{{k}} and the gradient statistics of Z10 reach the server without error, and the amplitude sum of the "
      f"U variants is exact except in the pilot ablation of Section 4.8. The channel is Rayleigh, either i.i.d. or Gauss–Markov with Jakes correlation, without path loss, shadowing or "
      f"correlation between antennas, and Jakes fading and eight antennas were tested only on the digits task; MNIST and CIFAR-10 used "
      f"i.i.d. channels and four antennas. The main task is a small linear model, the CNNs on MNIST and CIFAR-10 are small, and CIFAR-10 "
      f"reached only {BIG['cifar_ideal']:.0f}% in 300 rounds without noise, so the conclusions may not carry over to deep models "
      f"trained to convergence. The policies of [7], [10] and [12] are re-implementations with tuned or replaced constants; Y14 is our "
      f"own policy inspired by [14], whose full text we could not obtain; and Y4 uses bisection "
      f"instead of the linear scan of [4]. Thresholds, constants and the per-policy learning rates were tuned at one SNR on the same "
      f"channel family as the evaluation, and MNIST and CIFAR-10 used one learning rate for all policies. The unit-power convention "
      f"itself shrinks the aggregate to about {N['shrink_hi']:.2f} of the true gradient for all policies; the common scaling of "
      f"Section 4.8 removes this bias but costs accuracy at low SNR. Finally, MNIST and CIFAR-10 "
      f"used only five seeds at the reference SNR and three or five elsewhere, so their paired intervals are wide.")
    H("6. Conclusion", 1)
    P(f"We compared eleven power-control policies and four unit-gain variants for over-the-air federated learning under Rayleigh fading "
      f"on one benchmark, with unit-power symbols and paired confidence intervals. Full inversion failed as expected, and its "
      f"heavy-tailed MSE explains both its low accuracy and its seed variance. At a common learning rate a lower aggregation MSE did "
      f"not give a higher accuracy: the MSE-optimised policies OPT, Z10 and Y14 shrink the aggregated gradient, and with unit-gain "
      f"normalisation they reached "
      f"{min(av(m) for m in ('OPT-U', 'Z10-U', 'Y14-U')):.1f}–{max(av(m) for m in ('OPT-U', 'Z10-U', 'Y14-U')):.1f}% at −35 dB, "
      f"level with the capped inversion of [12] ({a1('C22')}) and ahead of M3 ({a1('M3')}) and M2 ({a1('M2')}). A learning rate "
      f"tuned per policy closed most of the same gap, so the problem of these policies is a bias that can be compensated. The advantage of "
      f"four-antenna truncation over single-antenna truncation is mostly array gain; refining the combiner helped clearly only with "
      f"eight antennas, "
      f"and the selection of [4] did not improve on a fixed combiner. Keeping every device with a capped amplitude was more robust to "
      f"label skew than truncation, and a threshold of 0.7 from a rule that needs only K was within "
      f"{max(r['acc_best'] - r['acc07'] for r in rule_rows()):.2f} points of the best grid value for K = 20, 50 and 100. "
      f"{BIG['conclusion']} Future work should test deeper models trained to convergence, measured channels, imperfect side "
      f"information, per-policy learning rates on the larger tasks, and the full algorithm of [14] with its convergence constants.")


def bA(g, m, s=-35.0):
    return bacc(g, (m, float(s)))[0]


def bP(g, m, f, s=-35.0):
    rows = g[(m, float(s))]
    if not rows:
        return float("nan")
    key = f if f in next(iter(rows.values())) else {"mse_mean": "mse"}.get(f, f)
    return float(np.mean([r[key] for r in rows.values()]))


def nseeds(g, m, s=-35.0):
    return len(g.get((m, float(s)), {}))


def larger(B):
    P, H, CAP, FIG, TAB = B.para, B.heading, B.caption, B.figure, B.table
    H("4.7 Larger tasks: MNIST and CIFAR-10", 2)
    g = MN
    u = ["OPT-U", "Z10-U", "Y14-U"]
    c_m3 = bci(g, "M3", "M2", -35.0); c_c12 = bci(g, "C22", "M2", -35.0); c_y = bci(g, "Y14-U", "Y14", -35.0)
    c_o = bci(g, "OPT-U", "OPT", -35.0); c_z = bci(g, "Z10-U", "Z10", -35.0)
    mse_rank = sorted((bP(g, m, "mse_mean"), m) for m in ["M1", "M2", "M3", "OPT", "OPT-U", "RCI", "C22", "Z10", "Z10-U", "Y14", "Y14-U"])
    assert {m for _, m in mse_rank[:3]} == {"OPT", "Z10", "Y14"}, mse_rank
    m1 = sorted(r["acc3"] for r in g[("M1", -35.0)].values())
    n_fail = sum(a < 20 for a in m1)
    assert 0 < n_fail < len(m1) and c_y["lo"] > 0 and c_m3["lo"] > 0
    assert min(("M2", "M3", "C22", "OPT-U"), key=lambda m: bA(g, m, -45)) == "M2"
    lo45 = min(bA(g, m, -45) for m in ("M2", "M3", "C22", "OPT-U")); hi45 = max(bA(g, m, -45) for m in ("M2", "M3", "C22", "OPT-U"))
    P(f"Table 10 gives the MNIST results with the 13,610-parameter CNN and K = 100 devices. At −35 dB the noiseless reference reached "
      f"{bA(g, 'IDEAL'):.1f}%, M2 {bA(g, 'M2'):.1f}%, M3 {bA(g, 'M3'):.1f}% and C12 {bA(g, 'C22'):.1f}%. M1 averaged "
      f"{bA(g, 'M1'):.1f}%: {words(len(m1) - n_fail)} seeds reached {m1[n_fail]:.0f}–{m1[-1]:.0f}% and the other {words(n_fail)} "
      f"stayed at chance level, which is the heavy tail of Section 4.4 again. OPT, Z10 and Y14 had the lowest MSE, their "
      f"shrinkage factors were {bP(g, 'OPT', 'proj'):.2f}, {bP(g, 'Z10', 'proj'):.2f} and {bP(g, 'Y14', 'proj'):.2f}, and Y14 "
      f"reached only {bA(g, 'Y14'):.1f}%. With unit-gain normalisation the three reached {min(bA(g, m) for m in u):.1f}–"
      f"{max(bA(g, m) for m in u):.1f}%. Normalisation helped Y14 most, by {ci_txt(c_y)}, Z10 by "
      f"{ci_txt(c_z)} and OPT by {ci_txt(c_o)}. With five seeds most intervals are wide: M3 − M2 was {ci_txt(c_m3)} and "
      f"C12 − M2 {ci_txt(c_c12)}. At −45 dB (three seeds) M2, M3, C12 and OPT-U reached {lo45:.1f}–{hi45:.1f}%, with M2 lowest.")
    rows = [["Policy", "−45 dB", "−40 dB", "−35 dB", "−30 dB", "MSE −35 dB", "Shrink­age −35 dB"]]
    for m in ["IDEAL", "M1", "M2", "M3", "OPT", "OPT-U", "RCI", "C22", "Z10", "Z10-U", "Y14", "Y14-U"]:
        rows.append(["Ideal" if m == "IDEAL" else nm(m)] + [bcell(g, (m, float(s))) for s in (-45, -40, -35, -30)] +
                    (["–", "1.00"] if m == "IDEAL" else [num(bP(g, m, "mse_mean"), 2 if bP(g, m, "mse_mean") < 100 else 0), num(bP(g, m, "proj"))]))
    CAP("Table 10.", "MNIST with a 13,610-parameter CNN and K = 100 devices: accuracy (%) as mean ± standard deviation over five seeds at "
        "−35 dB and three seeds at the other SNRs (mean of the evaluations after rounds 80, 90 and 100), with MSE and shrinkage factor "
        "at −35 dB. A dash marks configurations that were not run.")
    TAB(rows, [1250, 1250, 1250, 1250, 1250, 1100, 1100], sz=16)
    g = CF
    snrs = sorted({s for (m, s) in g})
    c_m3 = bci(g, "M3", "M2", -35.0); c_c12 = bci(g, "C22", "M2", -35.0)
    c_ou = bci(g, "OPT-U", "OPT", -35.0); c_yu = bci(g, "Y14-U", "Y14", -35.0); c_zu = bci(g, "Z10-U", "Z10", -35.0)
    c40 = bci(g, "C22", "M2", -40.0); m340 = bci(g, "M3", "M2", -40.0)
    m1 = [r for r in g[("M1", -35.0)].values()]
    assert all(not np.isfinite(r["mse_mean"]) for r in m1) and abs(bA(g, "M1") - 10.0) < 0.5
    assert min(c_ou["lo"], c_yu["lo"], c_zu["lo"]) > 0
    P(f"On CIFAR-10 the noiseless reference reached {bA(g, 'IDEAL'):.1f}% after 300 rounds (Table 11, Figure 10), a modest accuracy "
      f"for a small CNN trained with plain gradient steps on non-IID data, but enough to separate the policies. M1 diverged in all "
      f"{words(len(m1))} seeds: the model weights became non-finite early in training, and the accuracy stayed at chance level "
      f"({bA(g, 'M1'):.1f}%). At −35 dB M2 reached {bA(g, 'M2'):.1f}%, M3 {bA(g, 'M3'):.1f}% and C12 {bA(g, 'C22'):.1f}%; "
      f"M3 − M2 was {ci_txt(c_m3)}, C12 − M2 {ci_txt(c_c12)} and C12 − M3 {ci_txt(bci(g, 'C22', 'M3', -35.0))}.")
    P(f"The shrinkage effect was strongest on this task. OPT, Z10 and Y14 shrank the update to {bP(g, 'OPT', 'proj'):.2f}, "
      f"{bP(g, 'Z10', 'proj'):.2f} and {bP(g, 'Y14', 'proj'):.2f} of its length and reached {bA(g, 'OPT'):.1f}%, {bA(g, 'Z10'):.1f}% "
      f"and {bA(g, 'Y14'):.1f}%. Unit-gain normalisation raised them by {ci_txt(c_ou)}, {ci_txt(c_zu)} and {ci_txt(c_yu)}, to "
      f"{bA(g, 'OPT-U'):.1f}%, {bA(g, 'Z10-U'):.1f}% and {bA(g, 'Y14-U'):.1f}%. {CIFAR_EXTRA()}")
    rows = [["Policy"] + [f"{neg(int(s))} dB" for s in snrs] + ["MSE −35 dB", "Shrink­age −35 dB", "Part. −35 dB"]]
    for m in ["IDEAL", "M1", "M2", "M3", "OPT", "OPT-U", "C22", "Z10", "Z10-U", "Y14", "Y14-U"]:
        rows.append(["Ideal" if m == "IDEAL" else nm(m)] + [bcell(g, (m, s)) for s in snrs] +
                    (["–", "1.00", "1.00"] if m == "IDEAL" else [fin(bP(g, m, "mse_mean"), 2), fin(bP(g, m, "proj")), num(bP(g, m, "part"))]))
    CAP("Table 11.", "CIFAR-10 with a 136,874-parameter CNN and K = 50 devices: accuracy (%) as mean ± standard deviation over five "
        "seeds (mean of the evaluations after rounds 250, 275 and 300), with MSE, shrinkage factor and participation at −35 dB. The "
        "noiseless reference is listed under −35 dB. A dash marks configurations that were not run; M1 diverged, so its MSE and "
        "shrinkage factor are not defined.")
    TAB(rows, [1250] + [1300] * len(snrs) + [1100, 1050, 1000], sz=16)
    FIG(FIGS["cifar"], "Figure 10.", "CIFAR-10 test accuracy versus communication round at −35 dB (mean of five seeds, evaluated every "
        "25 rounds).", 12.5)


def fin(x, k=2):
    return num(x, k) if np.isfinite(x) else "–"


def neg(x):
    return str(x).replace("-", "−")


def CIFAR_EXTRA():
    g = CF
    four = ("M2", "M3", "C22", "OPT-U")
    c40, m40, o40 = bci(g, "C22", "M2", -40.0), bci(g, "M3", "M2", -40.0), bci(g, "OPT-U", "M2", -40.0)
    assert min(c40["lo"], m40["lo"], o40["lo"]) > 0
    d30 = [bci(g, x, y, -30.0) for i, x in enumerate(four) for y in four[i + 1:]]
    n_out = sum(not c["within_noise"] for c in d30)
    spread = max(bA(g, m, -30) for m in four) - min(bA(g, m, -30) for m in four)
    tail = ("no paired difference among them excluded zero" if n_out == 0 else
            f"{words(n_out)} of the {words(len(d30))} paired differences among them excluded zero")
    return (f"Keeping every device mattered most at low SNR. At −40 dB M2 fell to {bA(g, 'M2', -40):.1f}% and M3 to "
            f"{bA(g, 'M3', -40):.1f}%, while C12 kept {bA(g, 'C22', -40):.1f}% and OPT-U {bA(g, 'OPT-U', -40):.1f}%; "
            f"C12 − M2 was {ci_txt(c40)}. At −30 dB the four were within {spread:.1f} points of each other, and {tail}.")


BIG = {}
try:
    BIG["cifar_ideal"] = bA(CF, "IDEAL")
except Exception:
    BIG["cifar_ideal"] = float("nan")
BIG["conclusion"] = (f"The same pattern held on MNIST and was stronger on CIFAR-10, where unit-gain normalisation raised Z10 and Y14 by "
                     f"{bci(CF, 'Z10-U', 'Z10', -35.0)['diff']:.1f} and {bci(CF, 'Y14-U', 'Y14', -35.0)['diff']:.1f} points and the capped "
                     f"inversion kept {bA(CF, 'C22', -40):.1f}% at −40 dB, where truncation fell to {bA(CF, 'M2', -40):.1f}%.")
BIG_SENTENCE = "the shrinkage effect reappeared on MNIST and on CIFAR-10 with a 136,874-parameter CNN"
N["cifar_lr"] = 0.2
N["mnist_tau"] = 0.8
N["cifar_tau"] = 1.2
N["cifar_snr_txt"] = "−40, −35 and −30 dB"
TABLE2[-1][1] = "CNN, 136,874 parameters / 50 / 300 / 5 at −40, −35 and −30 dB"
ABSTRACT = _abstract()


# participation on the tuning rounds (Y4 against the combiner of M3) and class mix under label skew
def _tune_part():
    out = {"Y4": {}, "M3": {}}
    g = T.group(V("y4tune.jsonl"), by=("thr",))
    for (t,), rows in g.items():
        out["Y4"][t] = float(np.mean([r["part"] for r in rows.values()]))
    g = T.group(V("tune.jsonl"), by=("method", "thr"))
    for (m, t), rows in g.items():
        if m == "M3":
            out["M3"][t] = float(np.mean([r["part"] for r in rows.values()]))
    if 0.9 not in out["M3"]:
        out["M3"][0.9] = float("nan")
    return out


TP = _tune_part()


def _skew():
    import ota_fl_sim as S
    out = {}
    for a in (0.1, 0.5):
        cs = []
        for seed in range(10):
            Xtr, ytr, _, _ = S.load_data(seed)
            for p in S.partition(ytr, 50, a, np.random.default_rng(2000 + seed)):
                c = np.bincount(ytr[p], minlength=10)
                cs.append(int(((c / c.sum()) >= 0.1).sum()))
        out[a] = float(np.median(cs))
    return out


SKEW = _skew()
