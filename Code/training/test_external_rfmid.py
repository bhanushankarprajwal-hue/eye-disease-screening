"""
External test: how do the 8-class models do on RFMiD (a different dataset, different cameras/hospital)?
No retraining -- the models only ever saw ODIR images.

RFMiD is multi-label with its own disease names, so labels are mapped to ours:
  Normal = Disease_Risk 0 | Diabetes = DR | AMD = ARMD | Myopia = MYA | Hypertension = HR
  Glaucoma ~ ODC (optic disc cupping, a glaucoma sign -- proxy)
  Cataract ~ MH (media haze, often from cataract -- proxy)
  Others = abnormal image with none of the above
Two evaluations:
  1. Per-disease yes/no AUC on all images (multi-label ground truth).
  2. 8-class accuracy / macro-F1 on images that have exactly one mapped label (single-label subset).
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Code/
import predict_multiclass as pmc
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIR = os.environ.get("RFMID_DIR", os.path.join(ROOT, "external", "RFMiD", "test"))
lab = pd.read_csv(os.path.join(DIR, "labels.csv"), encoding="utf-8-sig")
lab["file"] = lab["ID"].astype(str) + ".jpg"
lab = lab[lab.file.apply(lambda f: os.path.isfile(os.path.join(DIR, f)))].reset_index(drop=True)
CLS = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Others"]
mapping = {"Diabetes": "DR", "Glaucoma": "ODC", "Cataract": "MH", "AMD": "ARMD", "Hypertension": "HR", "Myopia": "MYA"}
Y = pd.DataFrame({"Normal": (lab.Disease_Risk == 0).astype(int)})
for k, v in mapping.items():
    Y[k] = lab[v].astype(int)
Y["Others"] = ((lab.Disease_Risk == 1) & (Y[list(mapping)].sum(1) == 0)).astype(int)
print("RFMiD test images available:", len(lab))
print("Images per mapped label:", Y.sum().to_dict())

results = {}
import json as _j
for name, cfgfile in [("Stage 10 (EfficientNet-B0 448px)", "stage10_multiclass_config.json"),
                      ("Stage 12 ensemble", "stage12_multiclass_config.json")]:
    pmc.CONFIG_CANDIDATES = [os.path.join(ROOT, "Data", cfgfile)]
    cfg, m, d = pmc.load_multiclass_model()
    names = [cfg["class_names"][c] for c in cfg["classes"]]
    P = []
    for i, f in enumerate(lab.file):
        r = dict(pmc.predict_multiclass(cfg, m, d, [os.path.join(DIR, f)]))
        P.append([r[n] for n in CLS])
        if i % 100 == 0:
            print(f"  {name}: {i}/{len(lab)}", flush=True)
    P = np.array(P)
    aucs = {c: (roc_auc_score(Y[c], P[:, j]) if 0 < Y[c].sum() < len(Y) else np.nan) for j, c in enumerate(CLS)}
    abn_auc = roc_auc_score(1 - Y["Normal"], 1 - P[:, 0])
    single = Y.sum(1) == 1
    yt = Y[single].values.argmax(1); yp = P[single.values].argmax(1)
    present = sorted(set(yt))
    results[name] = {"auc": aucs, "normal_vs_abnormal_auc": abn_auc,
                     "single_label_n": int(single.sum()),
                     "single_label_accuracy": accuracy_score(yt, yp),
                     "single_label_macro_f1": f1_score(yt, yp, labels=present, average="macro", zero_division=0)}
    np.save(os.path.join(ROOT, "Data", f"rfmid_probs_{cfgfile.split('_')[0]}.npy"), P)

rows = []
for c in CLS:
    rows.append({"disease": c, "rfmid_label": mapping.get(c, "Disease_Risk=0" if c == "Normal" else "other diseases"),
                 "images": int(Y[c].sum()),
                 **{f"AUC {k}": round(v["auc"][c], 3) for k, v in results.items()}})
tab = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nPer-disease yes/no AUC on RFMiD test set:\n", tab.to_string(index=False))
summ = pd.DataFrame({k: {"Normal vs abnormal AUC": round(v["normal_vs_abnormal_auc"], 3),
                         "Mean AUC (8 classes)": round(float(np.nanmean(list(v["auc"].values()))), 3),
                         "Single-label images": v["single_label_n"],
                         "Single-label accuracy": round(v["single_label_accuracy"], 3),
                         "Single-label macro-F1": round(v["single_label_macro_f1"], 3)} for k, v in results.items()})
print("\n", summ.to_string())
tab.to_csv(os.path.join(ROOT, "Data", "external_rfmid_per_disease_auc.csv"), index=False)
summ.to_csv(os.path.join(ROOT, "Data", "external_rfmid_summary.csv"))
print("\nSaved Data/external_rfmid_per_disease_auc.csv and Data/external_rfmid_summary.csv")
