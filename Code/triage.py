"""
Batch screening / triage: checks many fundus photos at once and sorts them into four groups.

    1  REFER        -- the AI is fairly sure the eye shows a disease          -> see a doctor soon
    2  REVIEW       -- the AI is unsure                                       -> a doctor should look
    3  NORMAL       -- the AI is confident the eye looks normal               -> routine follow-up
    4  RETAKE       -- the photo itself is unusable (too dark, over-exposed,  -> take the photo again
                       almost no fundus, probably not a fundus photo)

The two decision limits (how sure is "fairly sure" / "confident") are NOT guessed: they are set on the
validation images by Code/calibrate_triage.py and saved in Data/triage_thresholds.json, so that only a
small, known share of diseased eyes can end up in the NORMAL group. Defaults are used until then.

Used by Code/app.py (the "Batch screening" mode). Also works from the terminal:

    python Code/triage.py --folder test_images
    python Code/triage.py --folder "D:/new_photos" --out Data/my_results.csv --sort-into "D:/sorted"
"""
import argparse
import io
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import image_quality as iq          # noqa: E402
import predict_multiclass as pmc    # noqa: E402

PROJECT_ROOT = pmc.PROJECT_ROOT
THRESHOLDS_FILE = os.path.join(PROJECT_ROOT, "Data", "triage_thresholds.json")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

GROUPS = {   # key: (order, short label, long label, advice)
    "refer":  (1, "Refer",          "Refer — likely disease",       "Send to an eye doctor soon."),
    "review": (2, "Doctor review",  "Doctor review — AI unsure",    "A doctor should look at this photo."),
    "normal": (3, "Likely normal",  "Likely normal",                "Routine follow-up."),
    "retake": (4, "Retake photo",   "Retake photo — unusable",      "Take the photo again; if it stays poor, a doctor should still check (a cataract can cause this)."),
}
DEFAULT_THRESHOLDS = {"refer_min_confidence": 0.60, "normal_min_confidence": 0.70}


def load_thresholds():
    """Returns (thresholds dict, info string). Calibrated values if Data/triage_thresholds.json exists."""
    t = dict(DEFAULT_THRESHOLDS)
    info = "default limits (not yet calibrated -- run Code/calibrate_triage.py)"
    if os.path.isfile(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE) as f:
            data = json.load(f)
        t.update(data.get("thresholds", {}))
        info = data.get("summary", "calibrated on validation images")
    return t, info


# ------------------------------------------------------------------ fast batched prediction
def _open(src):
    """src: a file path, raw bytes, or a file-like object."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    return pmc.load_rgb(src)


def predict_many(config, device, sources, batch_size=16, progress=None, workers=4):
    """8-class probabilities for many photos, in batches (much faster than one by one).
    Uses every network of the ensemble and mirrored copies, exactly like predict_multiclass.predict_multiclass.
    Returns (probs [N, 8] with NaN rows for unreadable files, list of error strings or None)."""
    members = config["_members"]
    n = len(sources)
    probs = np.full((n, len(config["classes"])), np.nan, dtype=np.float32)
    errors = [None] * n
    use_amp = device.type == "cuda"
    t0 = time.time()

    def prep(i):
        try:
            img = _open(sources[i])
            return i, [mb["transform"](img) for mb in members], None
        except Exception as e:  # noqa: BLE001 -- reported per file, the batch continues
            return i, None, f"could not read image ({e.__class__.__name__})"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, n, batch_size):
            idx = list(range(start, min(n, start + batch_size)))
            prepared = list(pool.map(prep, idx))
            ok = [(i, xs) for i, xs, err in prepared if xs is not None]
            for i, xs, err in prepared:
                if err:
                    errors[i] = err
            if ok:
                acc = None
                with torch.inference_mode(), torch.autocast(device.type, enabled=use_amp):
                    for m_i, mb in enumerate(members):
                        x = torch.stack([xs[m_i] for _, xs in ok]).to(device)
                        p = torch.softmax(mb["model"](x).float(), 1)
                        if mb["tta"]:
                            p = (p + torch.softmax(mb["model"](torch.flip(x, dims=[3])).float(), 1)) / 2
                        acc = p if acc is None else acc + p
                acc = (acc / len(members)).cpu().numpy()
                for row, (i, _) in enumerate(ok):
                    probs[i] = acc[row]
            if progress:
                done = idx[-1] + 1
                el = time.time() - t0
                progress(done, n, el / done * (n - done))
    return probs, errors


# ------------------------------------------------------------------ decision rule
def decide(prob_row, names, quality_status, t):
    """Returns (group key, reason). prob_row: 8 probabilities in the order of `names`."""
    if quality_status == "retake":
        return "retake", "photo quality check failed"
    if prob_row is None or np.isnan(prob_row).any():
        return "retake", "image could not be read"
    order = np.argsort(prob_row)[::-1]
    top, top_p = names[order[0]], float(prob_row[order[0]])
    p_normal = float(prob_row[names.index("Normal")])
    if top != "Normal" and top_p >= t["refer_min_confidence"]:
        return "refer", f"{top} {top_p:.0%} (at or above the {t['refer_min_confidence']:.0%} refer limit)"
    if top == "Normal" and p_normal >= t["normal_min_confidence"]:
        return "normal", f"Normal {p_normal:.0%} (at or above the {t['normal_min_confidence']:.0%} normal limit)"
    second = names[order[1]]
    return "review", f"not sure: {top} {top_p:.0%} vs {second} {float(prob_row[order[1]]):.0%}"


PATIENT_RE = re.compile(r"^(?P<pid>.+?)[ _\-\.]*(?P<eye>left|right|od|os|l|r)$", re.IGNORECASE)


def patient_and_eye(filename):
    """'123_left.jpg' -> ('123', 'Left'). Returns (filename stem, '') if no eye is named."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = PATIENT_RE.match(stem)
    if not m:
        return stem, ""
    eye = m.group("eye").lower()
    return m.group("pid"), "Left" if eye in ("left", "os", "l") else "Right"


def screen(config, device, items, batch_size=16, progress=None, thresholds=None):
    """items: list of (display_name, source) where source is a path or bytes.
    Returns a DataFrame with one row per photo, sorted by priority (most urgent first)."""
    t = thresholds or load_thresholds()[0]
    qt = iq.load_thresholds()
    names = [config["class_names"][c] for c in config["classes"]]
    sources = [s for _, s in items]

    def qcheck(s):
        try:
            return iq.assess(_open(s), qt)
        except Exception:  # noqa: BLE001
            return "retake", ["could not read image"], {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        quality = list(pool.map(qcheck, sources))
    probs, errors = predict_many(config, device, sources, batch_size=batch_size, progress=progress)

    rows = []
    for i, (name, _src) in enumerate(items):
        qs, qprob, _ = quality[i]
        p = probs[i]
        group, reason = decide(None if errors[i] else p, names, qs, t)
        pid, eye = patient_and_eye(name)
        row = {"file": name, "patient": pid, "eye": eye, "group": group,
               "group_label": GROUPS[group][1], "reason": reason,
               "quality": qs, "quality_notes": "; ".join(qprob) if qprob else ""}
        if not errors[i]:
            order = np.argsort(p)[::-1]
            row.update({"top_finding": names[order[0]], "confidence": float(p[order[0]]),
                        "second_finding": names[order[1]], "second_confidence": float(p[order[1]]),
                        "p_disease": float(1 - p[names.index("Normal")])})
            row.update({f"p_{n}": float(p[k]) for k, n in enumerate(names)})
        else:
            row.update({"top_finding": "", "confidence": np.nan, "second_finding": "",
                        "second_confidence": np.nan, "p_disease": np.nan})
        rows.append(row)
    df = pd.DataFrame(rows)
    return sort_by_priority(df)


def sort_by_priority(df):
    """Most urgent first: group order, then (inside REFER) highest disease confidence, (inside REVIEW)
    highest chance of disease, (inside NORMAL) lowest confidence first."""
    df = df.copy()
    df["_g"] = df.group.map(lambda g: GROUPS[g][0])
    key = np.where(df.group == "normal", df.confidence.fillna(0), -df.p_disease.fillna(0))
    key = np.where(df.group == "refer", -df.confidence.fillna(0), key)
    df["_k"] = key
    df = df.sort_values(["_g", "_k"]).drop(columns=["_g", "_k"]).reset_index(drop=True)
    df.insert(0, "priority", np.arange(1, len(df) + 1))
    return df


def per_patient(df):
    """One row per patient: the more urgent of the two eyes decides the patient's group."""
    if df.empty:
        return df
    d = df.copy()
    d["_g"] = d.group.map(lambda g: GROUPS[g][0] if g != "retake" else 2.5)  # retake ranks above normal
    rows = []
    for pid, sub in d.groupby("patient", sort=False):
        worst = sub.sort_values("_g").iloc[0]
        eyes = ", ".join(f"{(r.eye or r.file)}: {r.group_label}" + (f" ({r.top_finding} {r.confidence:.0%})" if r.top_finding else "")
                         for r in sub.itertuples())
        rows.append({"patient": pid, "photos": len(sub), "group": worst.group, "group_label": worst.group_label,
                     "main_finding": worst.top_finding, "confidence": worst.confidence,
                     "p_disease": worst.p_disease, "eyes": eyes})
    out = pd.DataFrame(rows)
    out["_g"] = out.group.map(lambda g: GROUPS[g][0])
    out = out.sort_values(["_g", "p_disease"], ascending=[True, False]).drop(columns="_g").reset_index(drop=True)
    out.insert(0, "priority", np.arange(1, len(out) + 1))
    return out


def list_images(folder):
    out = []
    for dp, _dn, fns in os.walk(folder):
        for f in sorted(fns):
            if f.lower().endswith(IMAGE_EXT):
                out.append(os.path.join(dp, f))
    return sorted(out)


def summary_text(df):
    c = df.group.value_counts()
    return ", ".join(f"{GROUPS[g][1]}: {int(c.get(g, 0))}" for g in GROUPS)


def main():
    ap = argparse.ArgumentParser(description="Screen a folder of fundus photos and sort them by urgency.")
    ap.add_argument("--folder", required=True, help="folder with fundus photos (sub-folders are included)")
    ap.add_argument("--out", default=None, help="CSV file for the results (default: <folder>/triage_results.csv)")
    ap.add_argument("--sort-into", default=None, help="optional: copy photos into 1_Refer/2_Doctor_review/... here")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    files = list_images(a.folder)
    if not files:
        sys.exit(f"No images found in {a.folder}")
    config, _m, device = pmc.load_multiclass_model()
    t, info = load_thresholds()
    print(f"Model: {pmc.describe(config)} | device: {device} | limits: {info}")
    print(f"Screening {len(files)} photos...")
    items = [(os.path.relpath(f, a.folder), f) for f in files]

    def progress(done, n, eta):
        print(f"\r  {done}/{n}  (~{eta:.0f}s left)   ", end="", flush=True)

    t0 = time.time()
    df = screen(config, device, items, batch_size=a.batch_size, progress=progress, thresholds=t)
    print(f"\nDone in {time.time()-t0:.1f}s. {summary_text(df)}")
    out = a.out or os.path.join(a.folder, "triage_results.csv")
    df.to_csv(out, index=False)
    print("Results saved to", out)
    if a.sort_into:
        for r in df.itertuples():
            g = GROUPS[r.group]
            d = os.path.join(a.sort_into, f"{g[0]}_{g[1].replace(' ', '_')}")
            os.makedirs(d, exist_ok=True)
            shutil.copy2(os.path.join(a.folder, r.file), os.path.join(d, f"{r.priority:04d}_{os.path.basename(r.file)}"))
        print("Photos copied into group folders under", a.sort_into)
    with pd.option_context("display.width", 160, "display.max_colwidth", 40):
        print(df[["priority", "file", "group_label", "top_finding", "confidence", "quality_notes"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
