"""
Stage 12 -- improving the 8-class model again (GPU).

What changes compared with Stage 10 (Code/train_stage10.py, 71.5% test accuracy):
  1. Longer training: up to 40 epochs (Stage 10 was still improving at its last epoch, 25).
  2. Bigger images: 512x512 (the full size of preprocessed_images/) instead of 448x448.
  3. Local contrast enhancement (Ben Graham's method, used by the winners of the Kaggle diabetic
     retinopathy competition): image*4 - blurred_image*4 + 128. Small spots and vessels stand out more.
  4. Ensemble: the new model's probabilities are averaged with the Stage 10 model's.
Same data and same patient-level split as Stage 9/10 (Data/stage9_multiclass_split.csv).
The final choice (Stage 10, Stage 12, or the ensemble) is made on the VALIDATION set only.

Run from the project root with venv311 active:
    python Code\\train_stage12.py
Everything printed is also saved to Data/stage12_training_log.txt
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
SMOKE = os.environ.get("MC_SMOKE") == "1"   # quick code test only (tiny data); leave unset for the real run
IMG_DIR = os.environ.get("MC_IMG_DIR", os.path.join(ROOT, "preprocessed_images"))
_log = open(os.path.join(ROOT, "Data", "stage12_training_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only")
if device.type != "cuda" and not SMOKE:
    print("WARNING: no GPU found -- this script is meant for the RTX 4060 and will be extremely slow on CPU.")

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {"N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract",
               "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Others"}
NAMES = [CLASS_NAMES[c] for c in CLASSES]
IMG = 128 if SMOKE else 512
BATCH = 4 if SMOKE else 6          # 6 fits an 8 GB laptop GPU at 512x512 with mixed precision
EPOCHS = 1 if SMOKE else 40
WARMUP_EPOCHS = 0 if SMOKE else 2
PATIENCE = 10
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ------------------------------------------------------------------ data
split = pd.read_csv(os.path.join("Data", "stage9_multiclass_split.csv"))
split["y"] = split["diagnosis"].map({c: i for i, c in enumerate(CLASSES)})
tr = split[split.split == "train"].reset_index(drop=True)
va = split[split.split == "val"].reset_index(drop=True)
te = split[split.split == "test"].reset_index(drop=True)
overlap = len(set(tr.ID) & set(va.ID)) + len(set(tr.ID) & set(te.ID)) + len(set(va.ID) & set(te.ID))
print("Images train/val/test:", len(tr), len(va), len(te), "| patients shared between splits:", overlap)
assert overlap == 0
if SMOKE:
    tr = tr.groupby("y").head(2).reset_index(drop=True)
    va = va.groupby("y").head(1).reset_index(drop=True)
    te = te.groupby("y").head(1).reset_index(drop=True)
    print("SMOKE mode -- tiny subset:", len(tr), len(va), len(te))


def load_rgb(path):
    """Open as RGB; swap channels back if the image is clearly blue (BGR-saved), as in predict_multiclass.py."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    if a[..., 2].mean() > a[..., 0].mean() + 10:
        img = Image.fromarray(np.ascontiguousarray(a[..., ::-1]))
    return img


def enhance(img, size):
    """Ben Graham local contrast enhancement on a size x size image; area outside the fundus circle set to grey."""
    img = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=size / 30))).astype(np.float32)
    out = np.clip(4 * a - 4 * blur + 128, 0, 255)
    yy, xx = np.ogrid[:size, :size]
    outside = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 > (0.95 * size / 2) ** 2
    out[outside] = 128
    return Image.fromarray(out.astype(np.uint8))


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, d, tf, size, enh):
        self.f, self.y, self.tf, self.size, self.enh = d.filename.tolist(), d.y.tolist(), tf, size, enh
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        img = load_rgb(os.path.join(IMG_DIR, self.f[i]))
        img = enhance(img, self.size) if self.enh else img.resize((self.size, self.size), Image.BILINEAR)
        return self.tf(img), self.y[i]


pin = device.type == "cuda"
train_dl = DataLoader(DS(tr, train_tf, IMG, True), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=pin)
val_dl = DataLoader(DS(va, eval_tf, IMG, True), batch_size=16, shuffle=False, num_workers=0, pin_memory=pin)
test_dl = DataLoader(DS(te, eval_tf, IMG, True), batch_size=16, shuffle=False, num_workers=0, pin_memory=pin)
print("Image size:", IMG, "| batch:", BATCH, "| batches per epoch:", len(train_dl), "| max epochs:", EPOCHS)

# ------------------------------------------------------------------ model
from torchvision.models import EfficientNet_B0_Weights
try:
    net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    pretrained = True
except Exception as e:
    if not SMOKE:
        raise
    print("SMOKE mode: ImageNet weights not downloadable here, random weights:", e)
    net = models.efficientnet_b0(weights=None); pretrained = False
net.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(net.classifier[1].in_features, len(CLASSES)))
net = net.to(device)
print("ImageNet pre-trained:", pretrained)

counts = np.bincount(tr.y, minlength=len(CLASSES)).astype(float)
w = 1.0 / np.sqrt(np.maximum(counts, 1)); w = w / w.mean()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device), label_smoothing=0.1)
optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
steps_total = EPOCHS * len(train_dl); warm = max(1, WARMUP_EPOCHS * len(train_dl))
sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / warm if s < warm else
                                          0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             classification_report, confusion_matrix, roc_auc_score)


def predict(model, dl, tta=True):
    model.eval(); P, Y = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            p = torch.softmax(model(x).float(), 1)
            if tta:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
            P.append(p.cpu()); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


CKPT = os.path.join("Code", "models", "stage12_multiclass_efficientnet_b0_512_enh.pt")
if SMOKE:
    CKPT = CKPT.replace(".pt", "_SMOKE.pt")
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
    vp, vy = predict(net, val_dl, tta=False)
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
print(f"Training time: {(time.time()-t0)/60:.1f} min | best val macro-F1 (no TTA): {best:.3f}")

h = pd.DataFrame(hist)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(h.epoch, h.train_loss, marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch")
ax[1].plot(h.epoch, h.val_acc, marker="o", label="val accuracy"); ax[1].plot(h.epoch, h.val_f1_macro, marker="o", label="val macro-F1")
ax[1].set_title("Validation"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join("Code", "notebooks", "stage12_training_curves.png"), dpi=150); plt.close()

# ------------------------------------------------------------------ evaluation: Stage 12, Stage 10, ensemble
net.load_state_dict(torch.load(CKPT, map_location=device))
v12, vy = predict(net, val_dl); t12, ty = predict(net, test_dl)

cfg10 = json.load(open(os.path.join("Data", "stage10_multiclass_config.json")))
m10 = models.efficientnet_b0(weights=None)
m10.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m10.classifier[1].in_features, len(CLASSES)))
m10.load_state_dict(torch.load(cfg10["checkpoint"], map_location=device)); m10 = m10.to(device)
S10 = 128 if SMOKE else cfg10.get("img_size", 448)
v10, _ = predict(m10, DataLoader(DS(va, eval_tf, S10, False), batch_size=16, num_workers=0))
t10, _ = predict(m10, DataLoader(DS(te, eval_tf, S10, False), batch_size=16, num_workers=0))
vens, tens = (v10 + v12) / 2, (t10 + t12) / 2


def metrics(y, p):
    return {"accuracy": accuracy_score(y, p),
            "precision_macro": precision_score(y, p, average="macro", zero_division=0),
            "recall_macro": recall_score(y, p, average="macro", zero_division=0),
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, p, average="weighted", zero_division=0)}


cands = {
    "Stage 10 - EfficientNet-B0, 448px": (v10, t10),
    "Stage 12 - EfficientNet-B0, 512px + contrast enhancement": (v12, t12),
    "Stage 12 ensemble - average of Stage 10 and Stage 12": (vens, tens),
}
rows = []
for name, (vp, tp) in cands.items():
    rows.append({"model": name, "val_f1_macro": f1_score(vy, vp.argmax(1), average="macro", zero_division=0),
                 **metrics(ty, tp.argmax(1)),
                 "mean_auc": float(np.nanmean([roc_auc_score((ty == i).astype(int), tp[:, i])
                                               if 0 < (ty == i).sum() < len(ty) else np.nan for i in range(len(CLASSES))]))})
comp = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print("\n" + comp.round(3).to_string())
winner = comp["val_f1_macro"].idxmax()
print("\nBest on VALIDATION macro-F1:", winner)

# the new config describes the better of {Stage 12 alone, ensemble}; predict_multiclass.py then compares it
# with Stage 10 by validation macro-F1 and uses whichever is higher.
new_name = max(list(cands)[1:], key=lambda k: comp.loc[k, "val_f1_macro"])
vp, tp = cands[new_name]; pred = tp.argmax(1)
print(f"\nDetails for: {new_name}\n")
print(classification_report(ty, pred, labels=range(len(CLASSES)), target_names=NAMES, zero_division=0))

cm = confusion_matrix(ty, pred, labels=range(len(CLASSES))); cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7.5, 6.5)); im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(8)); ax.set_xticklabels(NAMES, rotation=35, ha="right"); ax.set_yticks(range(8)); ax.set_yticklabels(NAMES)
for i in range(8):
    for j in range(8):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{new_name}\ntest confusion matrix", fontsize=10)
plt.colorbar(im, fraction=0.046); plt.tight_layout()
plt.savefig(os.path.join("Code", "notebooks", "stage12_confusion_matrix.png"), dpi=150); plt.close()

ovr = []
for i, nm in enumerate(NAMES):
    yt, yp = (ty == i).astype(int), (pred == i).astype(int)
    tp_, fn = int(((yt == 1) & (yp == 1)).sum()), int(((yt == 1) & (yp == 0)).sum())
    tn, fp = int(((yt == 0) & (yp == 0)).sum()), int(((yt == 0) & (yp == 1)).sum())
    ovr.append({"disease": nm, "test_images": int(yt.sum()),
                "auc": roc_auc_score(yt, tp[:, i]) if 0 < yt.sum() < len(yt) else float("nan"),
                "sensitivity": tp_ / max(tp_ + fn, 1), "specificity": tn / max(tn + fp, 1)})
ovr = pd.DataFrame(ovr).round(3)
print(ovr.to_string(index=False)); print("Mean AUC:", round(ovr.auc.mean(), 3))

if not SMOKE:
    comp.to_csv(os.path.join("Data", "stage12_comparison.csv"))
    ovr.to_csv(os.path.join("Data", "stage12_per_disease_yes_no.csv"), index=False)
    pd.DataFrame({"class": NAMES, "test_images": np.bincount(ty, minlength=8), "train_images": counts.astype(int),
                  "cnn_recall": recall_score(ty, pred, average=None, labels=range(8), zero_division=0),
                  "cnn_precision": precision_score(ty, pred, average=None, labels=range(8), zero_division=0),
                  "cnn_f1": f1_score(ty, pred, average=None, labels=range(8), zero_division=0)}).round(3) \
        .to_csv(os.path.join("Data", "stage12_per_class.csv"), index=False)
    m12 = {"architecture": "efficientnet_b0", "img_size": IMG, "preprocess": "ben_graham", "tta": True,
           "checkpoint": CKPT.replace("\\", "/")}
    m10c = {"architecture": "efficientnet_b0", "img_size": cfg10.get("img_size", 448), "preprocess": "none",
            "tta": True, "checkpoint": cfg10["checkpoint"]}
    is_ens = "ensemble" in new_name
    cfg = {"name": new_name, "classes": CLASSES, "class_names": CLASS_NAMES,
           "members": [m12, m10c] if is_ens else [m12],
           **({} if is_ens else m12),
           "best_val_f1_macro": float(comp.loc[new_name, "val_f1_macro"]),
           "per_class_csv": "Data/stage12_per_class.csv", "epochs_run": len(hist), "device_used": str(device),
           "test_metrics": {k: float(v) for k, v in comp.loc[new_name].items() if k != "val_f1_macro"}}
    if is_ens:
        cfg["checkpoint"] = m12["checkpoint"]  # used for the Grad-CAM heatmap
    json.dump(cfg, open(os.path.join("Data", "stage12_multiclass_config.json"), "w"), indent=2)
    print("\nSaved:", CKPT, "and Data/stage12_*.csv/json, Code/notebooks/stage12_*.png")
print("\nDONE -- Stage 12 finished.")
sys.stdout = sys.__stdout__; _log.close()
