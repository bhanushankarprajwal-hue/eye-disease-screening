"""
Calibrates and checks Code/image_quality.py on real photos (numpy + Pillow only; ~10 minutes, no GPU).

Ground truth for "bad photo": ODIR-5K's own diagnostic keywords. 30 raw ODIR photos are marked by the
graders as 'low image quality', 'optic disk photographically invisible', 'image offset',
'anterior segment image' or 'no fundus image'. All other ODIR raw photos are treated as usable.

Thresholds are set from USABLE photos only -- ODIR raw photos not marked bad, plus RFMiD train/val photos --
so that only ~0.5% of usable photos are sent back for a retake and ~5% get a soft warning. The 30 bad
photos and the RFMiD test photos (never used to set limits) then show how the check behaves.

Writes Data/quality_measurements.csv, Data/quality_thresholds.json, Data/quality_check_results.csv.
Usage:  python Code/calibrate_quality.py
"""
import json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Code/
import image_quality as iq

ROOT = iq.PROJECT_ROOT
RAW = os.path.join(ROOT, "ODIR-5K", "ODIR-5K", "Training Images")
XLSX = os.path.join(ROOT, "ODIR-5K", "ODIR-5K", "data.xlsx")
RF = os.path.join(ROOT, "external", "RFMiD")
OUT_M = os.path.join(ROOT, "Data", "quality_measurements.csv")
BAD_KEYS = ["low image quality", "optic disk photographically invisible", "image offset",
            "anterior segment image", "no fundus image"]


# The same 30 photos, listed here so the script also works without the openpyxl package (needed to read .xlsx)
KNOWN_BAD = {
    "372_left.jpg": "low image quality", "494_left.jpg": "optic disk photographically invisible",
    "1243_left.jpg": "image offset", "1254_left.jpg": "optic disk photographically invisible",
    "1319_left.jpg": "optic disk photographically invisible", "1706_left.jpg": "anterior segment image",
    "3935_left.jpg": "low image quality", "4007_left.jpg": "low image quality", "4149_left.jpg": "low image quality",
    "4169_left.jpg": "low image quality", "4290_left.jpg": "low image quality", "4442_left.jpg": "low image quality",
    "4448_left.jpg": "low image quality", "4522_left.jpg": "low image quality", "4580_left.jpg": "no fundus image",
    "372_right.jpg": "low image quality", "1228_right.jpg": "optic disk photographically invisible",
    "1369_right.jpg": "optic disk photographically invisible", "1710_right.jpg": "anterior segment image",
    "3947_right.jpg": "low image quality", "4066_right.jpg": "low image quality", "4124_right.jpg": "low image quality",
    "4149_right.jpg": "low image quality", "4180_right.jpg": "low image quality", "4198_right.jpg": "low image quality",
    "4262_right.jpg": "low image quality", "4319_right.jpg": "low image quality", "4394_right.jpg": "low image quality",
    "4427_right.jpg": "low image quality", "4601_right.jpg": "low image quality",
}


def bad_list():
    try:
        x = pd.read_excel(XLSX)
    except Exception:  # noqa: BLE001 -- openpyxl missing: use the list read from data.xlsx earlier
        return dict(KNOWN_BAD)
    bad = {}
    for fc, kc in [("Left-Fundus", "Left-Diagnostic Keywords"), ("Right-Fundus", "Right-Diagnostic Keywords")]:
        for f, k in zip(x[fc], x[kc]):
            for kk in BAD_KEYS:
                if kk in str(k):
                    bad[f] = kk
    return bad


T_START = time.time()
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "inf"))   # lets the measuring be split into several runs


def measure_all(files, source, done):
    rows, t0 = [], time.time()
    todo = [f for f in files if (source, os.path.basename(f)) not in done]
    for i, f in enumerate(todo):
        if time.time() - T_START > MAX_SECONDS:
            break
        try:
            m = iq.measure(f)
        except Exception as e:  # noqa: BLE001
            print("  could not read", f, e)
            continue
        rows.append({"source": source, "file": os.path.basename(f), **m})
        if (i + 1) % 1000 == 0:
            print(f"  {source}: {i+1}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    bad = bad_list()
    print("ODIR photos marked as bad quality by graders:", len(bad))
    old = pd.read_csv(OUT_M) if os.path.isfile(OUT_M) else pd.DataFrame(columns=["source", "file"])
    if len(old) and "corner" not in old.columns:      # file from an older version of the check -> start again
        old = pd.DataFrame(columns=["source", "file"])
    done = set(zip(old.source, old.file))
    rows = []
    rows += measure_all(sorted(os.path.join(RAW, f) for f in os.listdir(RAW) if f.lower().endswith(".jpg")), "odir_raw", done)
    for s in ["train", "val", "test"]:
        d = os.path.join(RF, s)
        if os.path.isdir(d):
            rows += measure_all(sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg")), "rfmid_" + s, done)
    m = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    m.to_csv(OUT_M, index=False)
    n_total = len(os.listdir(RAW)) + sum(len([f for f in os.listdir(os.path.join(RF, s)) if f.endswith(".jpg")])
                                         for s in ["train", "val", "test"] if os.path.isdir(os.path.join(RF, s)))
    if len(m) < n_total:
        print(f"Measured {len(m)}/{n_total} photos so far -- run again to continue.")
        sys.exit(0)
    m["bad"] = (m.source == "odir_raw") & m.file.isin(bad)
    m["reason"] = m.file.map(bad).where(m.bad)

    # usable photos used to SET the limits: ODIR raw (not marked bad) + RFMiD train/val; RFMiD test is kept apart
    good = m[((m.source == "odir_raw") & ~m.bad) | m.source.isin(["rfmid_train", "rfmid_val"])]
    q = lambda col, p: float(np.percentile(good[col], p))
    th = {
        "min_fundus_area": min(q("fundus_area", 0.5), 0.15),
        "min_brightness": q("brightness", 0.5),
        "max_saturated": max(q("saturated", 99.5), 0.05),
        "min_redness": q("redness", 0.5),
        "max_corner": max(q("corner", 99.5), 60.0),
        "min_detail": q("detail", 5),
        "max_uneven": q("uneven", 95),
    }
    th = {k: round(v, 4) for k, v in th.items()}
    with open(iq.THRESHOLDS_FILE, "w") as f:
        json.dump({"thresholds": th, "calibrated_on": f"{len(good)} usable photos (ODIR raw + RFMiD train/val)",
                   "rule": "retake limits = 0.5th/99.5th percentile of usable photos; warning limits = 5th/95th"}, f, indent=2)
    print("\nThresholds:", th)

    def status(row):
        s, probs, _ = iq_assess_from_row(row, th)
        return pd.Series({"status": s, "problems": "; ".join(probs)})

    def iq_assess_from_row(row, t):
        retake, warn = [], []
        if row.fundus_area < t["min_fundus_area"]:
            retake.append("too dark / almost no fundus visible")
        else:
            if row.brightness < t["min_brightness"]: retake.append("too dark")
            if row.saturated > t["max_saturated"]: retake.append("over-exposed (too bright)")
            if row.redness < t["min_redness"] or row.corner > t["max_corner"]: retake.append("may not be a fundus photo")
            if row.detail < t["min_detail"]: warn.append("blurry / low detail")
            if row.uneven > t["max_uneven"]: warn.append("uneven lighting")
        return ("retake" if retake else "warning" if warn else "ok"), retake + warn, None

    m = pd.concat([m, m.apply(status, axis=1)], axis=1)
    res = []
    for name, sub in [("ODIR usable photos", m[(m.source == "odir_raw") & ~m.bad]),
                      ("ODIR photos graders marked bad", m[m.bad]),
                      ("RFMiD test photos (not used to set limits)", m[m.source == "rfmid_test"])]:
        n = len(sub)
        res.append({"group": name, "photos": n,
                    "retake": int((sub.status == "retake").sum()),
                    "warning": int((sub.status == "warning").sum()),
                    "ok": int((sub.status == "ok").sum()),
                    "retake %": round(100 * (sub.status == "retake").mean(), 1),
                    "retake or warning %": round(100 * (sub.status != "ok").mean(), 1)})
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(ROOT, "Data", "quality_check_results.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n", res.to_string(index=False))
    print("\nBad photos, one by one:")
    print(m[m.bad][["file", "reason", "status", "problems"]].to_string(index=False))
    print("\nDONE -- quality check calibrated.")
