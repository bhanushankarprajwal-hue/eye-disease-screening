"""
Can the benchmark networks improve our final model? (ensemble search, no training)

Our final model = Stage 13 + Stage 12 (two EfficientNet-B0). Code/benchmark_models.py also trained DenseNet-121 and
ConvNeXt-Tiny on the same patient-level split. This script averages their probabilities in a few combinations and
chooses ONLY on the validation images (ODIR val + RFMiD val, the same rule used for Stage 13). The test sets
(ODIR test 960, RFMiD test 640) are only reported, never used to choose.
A new combination replaces the current model only if its validation macro-F1 is at least 0.005 higher
(smaller differences are noise).

Run from the project root with venv311 active (GPU, about 10 minutes):
    python Code\\ensemble_search.py
If the model changes, re-run Code\\calibrate_triage.py afterwards (the batch-screening limits depend on the model).
Outputs: Data/ensemble_search.csv, Data/ensemble_search_log.txt and, if better, Data/stage14_multiclass_config.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import predict_multiclass as pmc  # noqa: E402
import triage as tr  # noqa: E402

ROOT = pmc.PROJECT_ROOT
os.chdir(ROOT)
CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
NAMES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Others"]
IDX = {n: i for i, n in enumerate(NAMES)}
MIN_GAIN = 0.005
ODIR_DIR = os.path.join(ROOT, "preprocessed_images")
RF_DIR = os.path.join(ROOT, "external", "RFMiD")


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t); x.flush()
    def flush(self):
        for x in self.s: x.flush()


def rfmid_rows(part):
    """Same clear-label selection as Code/train_stage13.py."""
    d = os.path.join(RF_DIR, part)
    lab = pd.read_csv(os.path.join(d, "labels.csv"), encoding="utf-8-sig")
    lab["path"] = lab["ID"].astype(str).apply(lambda i: os.path.join(d, i + ".jpg"))
    lab = lab[lab.path.apply(os.path.isfile)]
    rows = []
    for _, r in lab.iterrows():
        rough = r.get("ODC", 0) + r.get("MH", 0) + r.get("HR", 0)
        clear = {"Diabetes": r["DR"], "AMD": r["ARMD"], "Myopia": r["MYA"]}
        if r["Disease_Risk"] == 0:
            rows.append((r.path, IDX["Normal"]))
        elif sum(clear.values()) == 1 and rough == 0:
            rows.append((r.path, IDX[[k for k, v in clear.items() if v == 1][0]]))
    return pd.DataFrame(rows, columns=["path", "y"])


def main():
    log = open(os.path.join("Data", "ensemble_search_log.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    cur = pmc.pick_config()
    print("Current model:", pmc.describe(cur))
    nets = {
        "S13": cur["members"][0],
        "S12": cur["members"][1],
        "DenseNet-121": {"architecture": "densenet121", "img_size": 384, "preprocess": "none", "tta": True,
                         "checkpoint": "Code/models/benchmark_densenet121.pt"},
        "ConvNeXt-Tiny": {"architecture": "convnext_tiny", "img_size": 384, "preprocess": "none", "tta": True,
                          "checkpoint": "Code/models/benchmark_convnext_tiny.pt"},
    }
    nets = {k: v for k, v in nets.items() if os.path.isfile(os.path.join(ROOT, v["checkpoint"]))}
    print("Networks available:", ", ".join(nets))

    sp = pd.read_csv(os.path.join("Data", "stage9_multiclass_split.csv"))
    sp["y"] = sp.diagnosis.map({c: i for i, c in enumerate(CLASSES)})
    sp["path"] = sp.filename.apply(lambda f: os.path.join(ODIR_DIR, f))
    val = pd.concat([sp[sp.split == "val"][["path", "y"]], rfmid_rows("val")], ignore_index=True)
    odir_te = sp[sp.split == "test"][["path", "y"]].reset_index(drop=True)
    rf = pd.read_csv(os.path.join(RF_DIR, "test", "labels.csv"), encoding="utf-8-sig")
    rf["path"] = rf["ID"].astype(str).apply(lambda i: os.path.join(RF_DIR, "test", i + ".jpg"))
    rf = rf[rf.path.apply(os.path.isfile)].reset_index(drop=True)
    mapping = {"Diabetes": "DR", "Glaucoma": "ODC", "Cataract": "MH", "AMD": "ARMD", "Hypertension": "HR", "Myopia": "MYA"}
    Y = pd.DataFrame({"Normal": (rf.Disease_Risk == 0).astype(int)})
    for k, v in mapping.items():
        Y[k] = rf[v].astype(int)
    Y["Others"] = ((rf.Disease_Risk == 1) & (Y[list(mapping)].sum(1) == 0)).astype(int)
    Y = Y[NAMES]
    print(f"Validation {len(val)} (ODIR + RFMiD) | ODIR test {len(odir_te)} | RFMiD test {len(rf)}")

    # probabilities of every single network on every image set (each network loaded once)
    P = {}
    for key, mc in nets.items():
        cfg = {"classes": CLASSES, "class_names": cur["class_names"], "members": [mc]}
        m = pmc.build_model(mc["architecture"], 8)
        m.load_state_dict(torch.load(os.path.join(ROOT, mc["checkpoint"]), map_location=device))
        m.to(device).eval()
        cfg["_members"] = [{"model": m, "transform": pmc._make_transform(mc["img_size"], mc.get("preprocess", "none")),
                            "tta": True, "img_size": mc["img_size"]}]
        P[key] = {}
        for s, df in (("val", val), ("odir", odir_te), ("rfmid", rf)):
            p, _ = tr.predict_many(cfg, device, df.path.tolist(), batch_size=16)
            P[key][s] = p
        print(f"  {key}: done")
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    combos = {"Current (S13 + S12)": ["S13", "S12"]}
    if "DenseNet-121" in nets:
        combos["S13 + S12 + DenseNet-121"] = ["S13", "S12", "DenseNet-121"]
        combos["S13 + DenseNet-121"] = ["S13", "DenseNet-121"]
    if "ConvNeXt-Tiny" in nets:
        combos["S13 + S12 + ConvNeXt-Tiny"] = ["S13", "S12", "ConvNeXt-Tiny"]
    if "DenseNet-121" in nets and "ConvNeXt-Tiny" in nets:
        combos["S13 + S12 + DenseNet-121 + ConvNeXt-Tiny"] = ["S13", "S12", "DenseNet-121", "ConvNeXt-Tiny"]

    def auc(y, p):
        return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan

    rows = []
    for name, keys in combos.items():
        pv = np.mean([P[k]["val"] for k in keys], 0)
        po = np.mean([P[k]["odir"] for k in keys], 0)
        pr = np.mean([P[k]["rfmid"] for k in keys], 0)
        ty = odir_te.y.values
        rows.append({"combination": name, "networks": len(keys),
                     "val_f1_macro (ODIR+RFMiD)": f1_score(val.y, pv.argmax(1), average="macro", zero_division=0),
                     "ODIR test accuracy": accuracy_score(ty, po.argmax(1)),
                     "ODIR test macro-F1": f1_score(ty, po.argmax(1), average="macro", zero_division=0),
                     "ODIR test mean AUC": float(np.nanmean([auc((ty == i).astype(int), po[:, i]) for i in range(8)])),
                     "RFMiD mean AUC": float(np.nanmean([auc(Y[c].values, pr[:, i]) for i, c in enumerate(NAMES)])),
                     "RFMiD AUC Myopia": auc(Y["Myopia"].values, pr[:, IDX["Myopia"]])})
    comp = pd.DataFrame(rows).set_index("combination")
    comp.to_csv(os.path.join("Data", "ensemble_search.csv"))
    pd.set_option("display.width", 250)
    print("\n" + comp.round(3).to_string())

    base = comp.loc["Current (S13 + S12)", "val_f1_macro (ODIR+RFMiD)"]
    best = comp["val_f1_macro (ODIR+RFMiD)"].idxmax()
    gain = comp.loc[best, "val_f1_macro (ODIR+RFMiD)"] - base
    print(f"\nBest on VALIDATION: {best} (gain {gain:+.4f} over the current model)")
    if best == "Current (S13 + S12)" or gain < MIN_GAIN:
        print(f"DECISION: keep the current model (no combination is at least {MIN_GAIN} better on validation).")
        print("DONE -- ensemble search finished.")
        return

    keys = combos[best]
    po = np.mean([P[k]["odir"] for k in keys], 0); ty = odir_te.y.values; pred = po.argmax(1)
    counts = pd.read_csv(cur["per_class_csv"]).train_images.values if cur.get("per_class_csv") else np.zeros(8)
    pd.DataFrame({"class": NAMES, "test_images": np.bincount(ty, minlength=8), "train_images": counts,
                  "cnn_recall": recall_score(ty, pred, average=None, labels=range(8), zero_division=0),
                  "cnn_precision": precision_score(ty, pred, average=None, labels=range(8), zero_division=0),
                  "cnn_f1": f1_score(ty, pred, average=None, labels=range(8), zero_division=0)}).round(3) \
        .to_csv(os.path.join("Data", "stage14_per_class.csv"), index=False)
    members = [nets[k] for k in keys]
    cfg = {"name": best.replace("S13", "Stage 13").replace("S12", "Stage 12") + " ensemble",
           "classes": CLASSES, "class_names": cur["class_names"], "members": members, **members[0],
           "selection": "highest macro-F1 on combined ODIR + RFMiD validation images (ensemble search, Stage 14)",
           "combined_val_f1_macro": float(comp.loc[best, "val_f1_macro (ODIR+RFMiD)"]),
           "per_class_csv": "Data/stage14_per_class.csv",
           "test_metrics": {"accuracy": float(comp.loc[best, "ODIR test accuracy"]),
                            "f1_macro": float(comp.loc[best, "ODIR test macro-F1"]),
                            "mean_auc": float(comp.loc[best, "ODIR test mean AUC"]),
                            "rfmid_mean_auc": float(comp.loc[best, "RFMiD mean AUC"])}}
    json.dump(cfg, open(os.path.join("Data", "stage14_multiclass_config.json"), "w"), indent=2)
    json.dump({"config": "Data/stage14_multiclass_config.json"}, open(os.path.join("Data", "multiclass_final_choice.json"), "w"))
    print(f"DECISION: the app now uses '{cfg['name']}'.")
    print("NEXT: run  python Code\\calibrate_triage.py  so the batch-screening limits match the new model.")
    print("DONE -- ensemble search finished.")


if __name__ == "__main__":
    main()
