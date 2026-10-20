"""
Stage 13 -- train the 8-class model on ODIR + RFMiD together (GPU).

Why: on RFMiD's test set (images from another hospital/camera, never seen in training) the Stage 10/12 models
dropped a lot for Myopia (AUC 0.995 on ODIR -> 0.76 on RFMiD). Training on images from both sources should
make the model work better on new hospitals' images.

Data
  ODIR   : same patient-level train/val/test split as Stages 9-12 (Data/stage9_multiclass_split.csv).
  RFMiD  : external/RFMiD/train and external/RFMiD/val are ADDED to training/validation.
           external/RFMiD/test is NOT used for training -- it stays an untouched outside test.
  RFMiD labels used (clear matches only):  Disease_Risk=0 -> Normal, DR -> Diabetes, ARMD -> AMD, MYA -> Myopia.
  An RFMiD image is used only if exactly one of DR/ARMD/MYA is set and it has no ODC/MH/HR label
  (those only roughly match Glaucoma/Cataract/Hypertension). Everything else in RFMiD is left out.
  Near-duplicate check: every RFMiD image is compared with every ODIR image (perceptual hash); matches are removed.

Model: EfficientNet-B0, 448x448, starting from the Stage 10 weights, fine-tuned on the mixed data.
Selection: Stage 10 vs Stage 12 ensemble vs Stage 13 vs (Stage 13 + Stage 12 ensemble), by macro-F1 on the
combined ODIR + RFMiD VALIDATION images. Test sets are only used for reporting.

Run from the project root with venv311 active:
    python Code\\train_stage13.py
Everything printed is also saved to Data/stage13_training_log.txt
"""
import os, sys, json, time, math, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t); x.flush()
    def flush(self):
        for x in self.s: x.flush()


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
SMOKE = os.environ.get("MC_SMOKE") == "1"
ODIR_DIR = os.environ.get("MC_IMG_DIR", os.path.join(ROOT, "preprocessed_images"))
RF_DIR = os.path.join(ROOT, "external", "RFMiD")
_log = open(os.path.join(ROOT, "Data", "stage13_training_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only")
CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {"N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract",
               "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Others"}
NAMES = [CLASS_NAMES[c] for c in CLASSES]
IDX = {n: i for i, n in enumerate(NAMES)}
IMG = 128 if SMOKE else 448
BATCH = 4 if SMOKE else 8
EPOCHS = 1 if SMOKE else 15
PATIENCE = 5
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    if a[..., 2].mean() > a[..., 0].mean() + 10:
        img = Image.fromarray(np.ascontiguousarray(a[..., ::-1]))
    return img


# ------------------------------------------------------------------ ODIR
split = pd.read_csv(os.path.join("Data", "stage9_multiclass_split.csv"))
split["y"] = split["diagnosis"].map({c: i for i, c in enumerate(CLASSES)})
split["path"] = split["filename"].apply(lambda f: os.path.join(ODIR_DIR, f))
split["source"] = "ODIR"
if SMOKE:
    split = split[split.path.apply(os.path.isfile)]
od = {s: split[split.split == s][["path", "y", "source"]].reset_index(drop=True) for s in ["train", "val", "test"]}


# ------------------------------------------------------------------ RFMiD
def rfmid_rows(part, strict=True):
    d = os.path.join(RF_DIR, part)
    if not os.path.isfile(os.path.join(d, "labels.csv")):
        return pd.DataFrame(columns=["path", "y", "source"])
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
    out = pd.DataFrame(rows, columns=["path", "y"]); out["source"] = "RFMiD"
    return out


rf_train, rf_val = rfmid_rows("train"), rfmid_rows("val")
print(f"RFMiD images used: train {len(rf_train)}, val {len(rf_val)}")
for nm, d in [("train", rf_train), ("val", rf_val)]:
    if len(d):
        print(f"  RFMiD {nm} per class:", {NAMES[k]: int(v) for k, v in d.y.value_counts().sort_index().items()})


# ------------------------------------------------------------------ near-duplicate check (difference hash)
def dhash(path, n=16):
    g = load_rgb(path).convert("L").resize((n + 1, n), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten())


if len(rf_train) + len(rf_val):
    t0 = time.time()
    od_paths = pd.concat([od["train"], od["val"], od["test"]]).path.tolist()
    od_h = np.stack([dhash(p) for p in od_paths])
    def is_dup(p):
        h = dhash(p)
        dist = np.unpackbits(np.bitwise_xor(od_h, h), axis=1).sum(1)
        return dist.min() <= 10   # <= 10 of 256 bits different = near-identical image
    for nm in ["rf_train", "rf_val"]:
        d = globals()[nm]
        dup = d.path.apply(is_dup)
        print(f"Near-duplicates of ODIR images found in {nm}: {int(dup.sum())} (removed)")
        globals()[nm] = d[~dup].reset_index(drop=True)
    print(f"Duplicate check took {time.time()-t0:.0f}s")

train_df = pd.concat([od["train"], rf_train], ignore_index=True)
val_df = pd.concat([od["val"], rf_val], ignore_index=True)
if SMOKE:
    train_df = train_df.groupby(["source", "y"]).head(2).reset_index(drop=True)
    val_df = val_df.groupby(["source", "y"]).head(1).reset_index(drop=True)
    od["test"] = od["test"].groupby("y").head(1).reset_index(drop=True)
print(f"Training images: {len(train_df)} (ODIR {int((train_df.source=='ODIR').sum())}, RFMiD {int((train_df.source=='RFMiD').sum())})")
print(f"Validation images: {len(val_df)} (ODIR {int((val_df.source=='ODIR').sum())}, RFMiD {int((val_df.source=='RFMiD').sum())})")


# ------------------------------------------------------------------ data loading
def enhance(img, size):
    img = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=size / 30))).astype(np.float32)
    out = np.clip(4 * a - 4 * blur + 128, 0, 255)
    yy, xx = np.ogrid[:size, :size]
    out[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 > (0.95 * size / 2) ** 2] = 128
    return Image.fromarray(out.astype(np.uint8))


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, paths, ys, tf, size, enh=False):
        self.p, self.y, self.tf, self.size, self.enh = list(paths), list(ys), tf, size, enh
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        img = load_rgb(self.p[i])
        img = enhance(img, self.size) if self.enh else img.resize((self.size, self.size), Image.BILINEAR)
        return self.tf(img), self.y[i]


pin = device.type == "cuda"
train_dl = DataLoader(DS(train_df.path, train_df.y, train_tf, IMG), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=pin)
val_dl = DataLoader(DS(val_df.path, val_df.y, eval_tf, IMG), batch_size=16, num_workers=0, pin_memory=pin)


def build_effnet():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, len(CLASSES)))
    return m


cfg10 = json.load(open(os.path.join("Data", "stage10_multiclass_config.json")))
net = build_effnet()
net.load_state_dict(torch.load(cfg10["checkpoint"], map_location="cpu"))
net = net.to(device)
print("Starting from the Stage 10 weights:", cfg10["checkpoint"])

counts = np.bincount(train_df.y, minlength=len(CLASSES)).astype(float)
w = 1.0 / np.sqrt(np.maximum(counts, 1)); w = w / w.mean()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device), label_smoothing=0.1)
optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
steps_total = EPOCHS * len(train_dl)
sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, steps_total))))
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report


def predict(model, dl, tta=True):
    model.eval(); P = []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            p = torch.softmax(model(x).float(), 1)
            if tta:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
            P.append(p.cpu())
    return torch.cat(P).numpy()


CKPT = os.path.join("Code", "models", "stage13_multiclass_efficientnet_b0_odir_rfmid.pt")
if SMOKE:
    CKPT = CKPT.replace(".pt", "_SMOKE.pt")
vy = val_df.y.values
hist, best, bad, t0 = [], -1.0, 0, time.time()
for ep in range(1, EPOCHS + 1):
    te0 = time.time(); net.train(); tot = n = 0
    for x, y in train_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(net(x), y)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); sched.step()
        tot += loss.item() * len(y); n += len(y)
    vp = predict(net, val_dl, tta=False)
    vf1 = f1_score(vy, vp.argmax(1), average="macro", zero_division=0); vacc = accuracy_score(vy, vp.argmax(1))
    hist.append({"epoch": ep, "train_loss": tot / n, "val_acc": vacc, "val_f1_macro": vf1})
    better = vf1 > best
    if better:
        best, bad = vf1, 0; torch.save(net.state_dict(), CKPT)
    else:
        bad += 1
    print(f"Epoch {ep:2d} | loss {tot/n:.4f} | val acc {vacc:.3f} | val macro-F1 {vf1:.3f}"
          f"{'  <- best, saved' if better else ''} | {time.time()-te0:.0f}s", flush=True)
    if bad >= PATIENCE:
        print(f"Early stopping: no validation improvement for {PATIENCE} epochs"); break
print(f"Training time: {(time.time()-t0)/60:.1f} min")
h = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(h.epoch, h.train_loss, marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch")
ax[1].plot(h.epoch, h.val_acc, marker="o", label="val accuracy"); ax[1].plot(h.epoch, h.val_f1_macro, marker="o", label="val macro-F1")
ax[1].set_title("Validation (ODIR + RFMiD)"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join("Code", "notebooks", "stage13_training_curves.png"), dpi=150); plt.close()
net.load_state_dict(torch.load(CKPT, map_location=device))

# ------------------------------------------------------------------ candidates on validation + both test sets
cfg12 = json.load(open(os.path.join("Data", "stage12_multiclass_config.json")))
m10 = build_effnet(); m10.load_state_dict(torch.load(cfg10["checkpoint"], map_location="cpu")); m10 = m10.to(device)
m12 = build_effnet(); m12.load_state_dict(torch.load(cfg12["members"][0]["checkpoint"], map_location="cpu")); m12 = m12.to(device)
S10, S12 = (128, 128) if SMOKE else (448, 512)

rf_test_lab = pd.read_csv(os.path.join(RF_DIR, "test", "labels.csv"), encoding="utf-8-sig")
rf_test_lab["path"] = rf_test_lab["ID"].astype(str).apply(lambda i: os.path.join(RF_DIR, "test", i + ".jpg"))
rf_test_lab = rf_test_lab[rf_test_lab.path.apply(os.path.isfile)].reset_index(drop=True)
if SMOKE:
    rf_test_lab = rf_test_lab.head(16)

sets = {"val": (val_df.path, vy), "odir_test": (od["test"].path, od["test"].y.values),
        "rfmid_test": (rf_test_lab.path, np.zeros(len(rf_test_lab), int))}


def probs_for(model, size, enh, paths):
    return predict(model, DataLoader(DS(paths, np.zeros(len(paths), int), eval_tf, size, enh), batch_size=16, num_workers=0))


P = {}
for s, (paths, _) in sets.items():
    p10 = probs_for(m10, S10, False, paths); p12 = probs_for(m12, S12, True, paths); p13 = probs_for(net, IMG, False, paths)
    P[s] = {"Stage 10": p10, "Stage 12 ensemble": (p10 + p12) / 2, "Stage 13 (ODIR+RFMiD)": p13,
            "Stage 13 + Stage 12 ensemble": (p13 + p12) / 2}

# RFMiD multi-label ground truth (same mapping as Code/test_external_rfmid.py)
mapping = {"Diabetes": "DR", "Glaucoma": "ODC", "Cataract": "MH", "AMD": "ARMD", "Hypertension": "HR", "Myopia": "MYA"}
Y = pd.DataFrame({"Normal": (rf_test_lab.Disease_Risk == 0).astype(int)})
for k, v in mapping.items():
    Y[k] = rf_test_lab[v].astype(int)
Y["Others"] = ((rf_test_lab.Disease_Risk == 1) & (Y[list(mapping)].sum(1) == 0)).astype(int)
Y = Y[NAMES]


def auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


rows = []
for name in P["val"]:
    pv, po, pr = P["val"][name], P["odir_test"][name], P["rfmid_test"][name]
    ty = sets["odir_test"][1]
    rf_auc = {c: auc(Y[c].values, pr[:, i]) for i, c in enumerate(NAMES)}
    rows.append({"model": name,
                 "val_f1_macro (ODIR+RFMiD)": f1_score(vy, pv.argmax(1), average="macro", zero_division=0),
                 "ODIR test accuracy": accuracy_score(ty, po.argmax(1)),
                 "ODIR test macro-F1": f1_score(ty, po.argmax(1), average="macro", zero_division=0),
                 "ODIR test mean AUC": float(np.nanmean([auc((ty == i).astype(int), po[:, i]) for i in range(8)])),
                 "RFMiD normal-vs-abnormal AUC": auc(1 - Y["Normal"].values, 1 - pr[:, IDX["Normal"]]),
                 "RFMiD mean AUC": float(np.nanmean(list(rf_auc.values()))),
                 **{f"RFMiD AUC {c}": rf_auc[c] for c in ["Diabetes", "AMD", "Myopia", "Glaucoma", "Cataract"]}})
comp = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 250)
print("\n" + comp.round(3).T.to_string())
winner = comp["val_f1_macro (ODIR+RFMiD)"].idxmax()
print("\nBest on VALIDATION (ODIR + RFMiD):", winner)

if not SMOKE:
    comp.to_csv(os.path.join("Data", "stage13_comparison.csv"))
    ty = sets["odir_test"][1]; pred = P["odir_test"][winner].argmax(1)
    print(f"\nODIR test, {winner}:\n")
    print(classification_report(ty, pred, labels=range(8), target_names=NAMES, zero_division=0))
    pd.DataFrame({"class": NAMES, "test_images": np.bincount(ty, minlength=8), "train_images": counts.astype(int),
                  "cnn_recall": recall_score(ty, pred, average=None, labels=range(8), zero_division=0),
                  "cnn_precision": precision_score(ty, pred, average=None, labels=range(8), zero_division=0),
                  "cnn_f1": f1_score(ty, pred, average=None, labels=range(8), zero_division=0)}).round(3) \
        .to_csv(os.path.join("Data", "stage13_per_class.csv"), index=False)
    m13c = {"architecture": "efficientnet_b0", "img_size": IMG, "preprocess": "none", "tta": True, "checkpoint": CKPT.replace("\\", "/")}
    m10c = {"architecture": "efficientnet_b0", "img_size": 448, "preprocess": "none", "tta": True, "checkpoint": cfg10["checkpoint"]}
    m12c = cfg12["members"][0]
    members = {"Stage 10": [m10c], "Stage 12 ensemble": [m12c, m10c], "Stage 13 (ODIR+RFMiD)": [m13c],
               "Stage 13 + Stage 12 ensemble": [m13c, m12c]}[winner]
    cfg = {"name": winner, "classes": CLASSES, "class_names": CLASS_NAMES, "members": members, **members[0],
           "selection": "highest macro-F1 on combined ODIR + RFMiD validation images (Stage 13)",
           "combined_val_f1_macro": float(comp.loc[winner, "val_f1_macro (ODIR+RFMiD)"]),
           "per_class_csv": "Data/stage13_per_class.csv", "epochs_run": len(hist), "device_used": str(device),
           "test_metrics": {"accuracy": float(comp.loc[winner, "ODIR test accuracy"]),
                            "f1_macro": float(comp.loc[winner, "ODIR test macro-F1"]),
                            "mean_auc": float(comp.loc[winner, "ODIR test mean AUC"]),
                            "rfmid_mean_auc": float(comp.loc[winner, "RFMiD mean AUC"])}}
    json.dump(cfg, open(os.path.join("Data", "stage13_multiclass_config.json"), "w"), indent=2)
    # Code/predict_multiclass.py reads this file first, so the app uses the Stage 13 selection.
    json.dump({"config": "Data/stage13_multiclass_config.json"}, open(os.path.join("Data", "multiclass_final_choice.json"), "w"))
    print("\nSaved:", CKPT, "Data/stage13_*.csv/json, Data/multiclass_final_choice.json, Code/notebooks/stage13_training_curves.png")
print("\nDONE -- Stage 13 finished.")
sys.stdout = sys.__stdout__; _log.close()
