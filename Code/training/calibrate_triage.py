"""
Sets and tests the two decision limits of the batch-screening tool (Code/triage.py).

  * "Likely normal" limit -- how sure the model must be that an eye is Normal. Chosen on the VALIDATION
    images so that at most 5% of diseased eyes can land in the "Likely normal" group (safety first).
  * "Refer" limit -- how sure the model must be about a disease. Chosen on the VALIDATION images so that
    at least 90% of the photos in the "Refer" group really show a disease.

The limits are then checked ONCE on the test images (ODIR test + RFMiD test), which were never used to
choose anything. Validation = ODIR val (960) + RFMiD val (640); test = ODIR test (960) + RFMiD test (640).

Writes: Data/triage_thresholds.json (read by the app), Data/triage_evaluation.csv,
        Data/triage_probs.csv (cached model outputs), Data/triage_calibration_log.txt,
        Code/notebooks/triage_evaluation.png
Usage (GPU recommended, ~5-10 min):   python Code/calibrate_triage.py
"""
import json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Code/
import predict_multiclass as pmc
import triage as tr
import image_quality as iq

ROOT = pmc.PROJECT_ROOT
os.chdir(ROOT)
MISS_TARGET = float(os.environ.get("TRIAGE_MISS_TARGET", "0.05"))     # max share of diseased eyes called "Likely normal"
REFER_PRECISION = float(os.environ.get("TRIAGE_REFER_PRECISION", "0.90"))
ODIR_DIR = os.path.join(ROOT, "preprocessed_images")
RF_DIR = os.path.join(ROOT, "external", "RFMiD")
CACHE = os.path.join(ROOT, "Data", "triage_probs.csv")


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t)
    def flush(self):
        for x in self.s: x.flush()


_log = open(os.path.join(ROOT, "Data", "triage_calibration_log.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log)

RF_MAP = {"Diabetes": "DR", "AMD": "ARMD", "Myopia": "MYA", "Hypertension": "HR", "Glaucoma": "ODC", "Cataract": "MH"}


def image_table():
    sp = pd.read_csv(os.path.join("Data", "stage9_multiclass_split.csv"))
    names = {"N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract", "A": "AMD", "H": "Hypertension",
             "M": "Myopia", "O": "Others"}
    rows = []
    for s in ["val", "test"]:
        for r in sp[sp.split == s].itertuples():
            p = os.path.join(ODIR_DIR, r.filename)
            if os.path.isfile(p):
                rows.append({"set": f"ODIR {s}", "split": s, "file": r.filename, "path": p,
                             "truth": names[r.diagnosis], "diseased": int(r.diagnosis != "N"), "rf_labels": ""})
    for s in ["val", "test"]:
        d = os.path.join(RF_DIR, s)
        lab_f = os.path.join(d, "labels.csv")
        if not os.path.isfile(lab_f):
            print(f"RFMiD {s} not found -- skipped")
            continue
        lab = pd.read_csv(lab_f, encoding="utf-8-sig")
        disease_cols = [c for c in lab.columns if c not in ("ID", "Disease_Risk")]
        for r in lab.itertuples(index=False):
            r = r._asdict()
            p = os.path.join(d, f"{r['ID']}.jpg")
            if not os.path.isfile(p):
                continue
            pos = [c for c in disease_cols if r[c] == 1]
            rows.append({"set": f"RFMiD {s}", "split": s, "file": f"{r['ID']}.jpg", "path": p,
                         "truth": "Normal" if r["Disease_Risk"] == 0 else "Abnormal",
                         "diseased": int(r["Disease_Risk"] == 1), "rf_labels": " ".join(pos)})
    return pd.DataFrame(rows)


def finding_correct(row):
    """Is the top finding one of the true diseases? (only meaningful for diseased eyes)"""
    if not row.diseased or not row.top_finding:
        return np.nan
    if row.set.startswith("ODIR"):
        return float(row.top_finding == row.truth)
    labels = set(str(row.rf_labels).split())
    if row.top_finding in RF_MAP:
        return float(RF_MAP[row.top_finding] in labels)
    if row.top_finding == "Others":
        return float(bool(labels - set(RF_MAP.values())))
    return 0.0


if __name__ == "__main__":
    config, _model, device = pmc.load_multiclass_model()
    names = [config["class_names"][c] for c in config["classes"]]
    model_name = pmc.describe(config)
    print("Model:", model_name, "| device:", device)
    imgs = image_table()
    if os.environ.get("TRIAGE_SMOKE") == "1":   # quick code check only
        imgs = imgs.groupby("set").head(40).reset_index(drop=True)
        CACHE = CACHE.replace(".csv", "_smoke.csv")
    print("Images:", imgs.groupby("set").size().to_dict())

    # ---------------------------------------------------------------- model outputs (cached)
    cached = pd.read_csv(CACHE) if os.path.isfile(CACHE) else None
    if cached is not None and cached.get("model", pd.Series([""])).iloc[0] == model_name and len(cached) == len(imgs):
        print("Using cached model outputs:", CACHE)
        P = cached[[f"p_{n}" for n in names]].values
        qstat = cached["quality"].tolist()
    else:
        t0 = time.time()
        P, err = tr.predict_many(config, device, imgs.path.tolist(), batch_size=16,
                                 progress=lambda d, n, eta: print(f"\r  {d}/{n} (~{eta:.0f}s left)  ", end="", flush=True))
        print(f"\nModel outputs in {time.time()-t0:.0f}s")
        qt = iq.load_thresholds()
        qstat = [iq.assess(p, qt)[0] for p in imgs.path]
        c = imgs[["set", "file"]].copy()
        c["model"] = model_name
        c["quality"] = qstat
        for k, n in enumerate(names):
            c[f"p_{n}"] = P[:, k]
        c.to_csv(CACHE, index=False)
    imgs["quality"] = qstat
    pn = P[:, names.index("Normal")]
    top = P.argmax(1)
    topp = P.max(1)
    is_normal_top = np.array([names[i] == "Normal" for i in top])
    usable = np.array([q != "retake" for q in qstat])

    # ---------------------------------------------------------------- choose limits on VALIDATION only
    val = (imgs.split == "val").values & usable
    dis = imgs.diseased.values == 1
    grid = np.round(np.arange(0.30, 0.991, 0.01), 2)
    t_normal = None
    for t in grid:   # smallest limit (= biggest "Likely normal" group) that keeps missed disease <= target
        missed = (val & dis & is_normal_top & (pn >= t)).sum() / max((val & dis).sum(), 1)
        if missed <= MISS_TARGET:
            t_normal = float(t)
            break
    t_normal = t_normal if t_normal is not None else 0.99
    t_refer = None
    for t in grid:   # smallest limit (= biggest "Refer" group) that keeps >= REFER_PRECISION really diseased
        sel = val & ~is_normal_top & (topp >= t)
        if sel.sum() >= 20 and dis[sel].mean() >= REFER_PRECISION:
            t_refer = float(t)
            break
    t_refer = t_refer if t_refer is not None else 0.99
    th = {"refer_min_confidence": t_refer, "normal_min_confidence": t_normal}
    print(f"\nLimits chosen on {val.sum()} usable validation images: refer >= {t_refer:.2f}, likely-normal >= {t_normal:.2f}")

    # ---------------------------------------------------------------- apply the rule to every image
    groups, tops = [], []
    for i in range(len(imgs)):
        g, _ = tr.decide(P[i], names, qstat[i], th)
        groups.append(g)
        tops.append(names[top[i]])
    imgs["group"] = groups
    imgs["top_finding"] = tops
    imgs["finding_correct"] = imgs.apply(finding_correct, axis=1)

    def evaluate(sub, label):
        out = []
        n_dis = int(sub.diseased.sum())
        n_healthy = int((sub.diseased == 0).sum())
        for g in tr.GROUPS:
            s = sub[sub.group == g]
            out.append({"images": label, "group": tr.GROUPS[g][1], "photos": len(s),
                        "share of photos %": round(100 * len(s) / max(len(sub), 1), 1),
                        "really diseased %": round(100 * s.diseased.mean(), 1) if len(s) else np.nan,
                        "top finding correct % (diseased)": round(100 * s.finding_correct.mean(), 1)
                        if s.finding_correct.notna().any() else np.nan})
        missed = int(((sub.group == "normal") & (sub.diseased == 1)).sum())
        cleared = int(((sub.group == "normal") & (sub.diseased == 0)).sum())
        out.append({"images": label, "group": "SUMMARY", "photos": len(sub),
                    "share of photos %": np.nan,
                    "really diseased %": round(100 * n_dis / max(len(sub), 1), 1),
                    "top finding correct % (diseased)": np.nan,
                    "diseased eyes wrongly in 'Likely normal' %": round(100 * missed / max(n_dis, 1), 1),
                    "diseased eyes sent to Refer or Review %": round(100 * ((sub.group.isin(['refer', 'review'])) & (sub.diseased == 1)).sum() / max(n_dis, 1), 1),
                    "healthy eyes cleared as 'Likely normal' %": round(100 * cleared / max(n_healthy, 1), 1)})
        return out

    rows = []
    for label, mask in [("VALIDATION (ODIR + RFMiD)", imgs.split == "val"),
                        ("TEST ODIR", imgs.set == "ODIR test"),
                        ("TEST RFMiD", imgs.set == "RFMiD test"),
                        ("TEST both", imgs.split == "test")]:
        rows += evaluate(imgs[mask], label)
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join("Data", "triage_evaluation.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print("\n", ev.to_string(index=False))

    te = imgs[imgs.split == "test"]
    missed = 100 * ((te.group == "normal") & (te.diseased == 1)).sum() / max(te.diseased.sum(), 1)
    summary = (f"Calibrated on {int((imgs.split == 'val').sum())} validation images (ODIR + RFMiD): refer if disease "
               f"confidence >= {t_refer:.0%}, likely normal if Normal confidence >= {t_normal:.0%}. On {len(te)} unseen "
               f"test images, {missed:.1f}% of diseased eyes fell in 'Likely normal'.")
    with open(tr.THRESHOLDS_FILE, "w") as f:
        json.dump({"thresholds": th, "summary": summary, "model": model_name,
                   "miss_target": MISS_TARGET, "refer_precision_target": REFER_PRECISION}, f, indent=2)
    print("\n" + summary)

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (label, mask) in zip(axes, [("ODIR test (960)", imgs.set == "ODIR test"), ("RFMiD test (640)", imgs.set == "RFMiD test")]):
        sub = imgs[mask]
        gl = [tr.GROUPS[g][1] for g in tr.GROUPS]
        dis_n = [int(((sub.group == g) & (sub.diseased == 1)).sum()) for g in tr.GROUPS]
        hea_n = [int(((sub.group == g) & (sub.diseased == 0)).sum()) for g in tr.GROUPS]
        ax.bar(gl, dis_n, color="#c0504d", label="really diseased")
        ax.bar(gl, hea_n, bottom=dis_n, color="#4f81bd", label="really healthy")
        for x, (a, b) in enumerate(zip(dis_n, hea_n)):
            ax.text(x, a + b + 3, str(a + b), ha="center", fontsize=9)
        ax.set_title(label); ax.set_ylabel("photos")
        ax.legend(frameon=False)
    fig.suptitle(f"Batch screening groups vs the truth (limits: refer >= {t_refer:.0%}, normal >= {t_normal:.0%})")
    fig.tight_layout()
    fig.savefig(os.path.join("Code", "notebooks", "triage_evaluation.png"), dpi=130)
    print("\nSaved Data/triage_thresholds.json, Data/triage_evaluation.csv, Code/notebooks/triage_evaluation.png")
    print("DONE -- triage calibrated.")
