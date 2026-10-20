"""
Stage 10 -- improved 8-class model (EfficientNet-B0, 448px, full fine-tune).
Same code as Code/notebooks/10_multiclass_improved.ipynb, as a plain script so it can be run from the terminal.

Run from the project root (Capstone Datasets folder) with venv311 active:
    python Code\\train_stage10.py

Everything printed is also saved to Data/stage10_training_log.txt
"""
import os, sys
import matplotlib
matplotlib.use("Agg")  # save charts to files, no pop-up windows

class _Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t); x.flush()
    def flush(self):
        for x in self.s: x.flush()

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_root)  # so the script works no matter which folder it is started from
_log = open(os.path.join(_root, "Data", "stage10_training_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log)


# ======================================================================
# Stage 10 — Improving the 8-class image model (GPU)
# ======================================================================

# ======================================================================
# 1. Setup
# ======================================================================
import os, json, time, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

CANDIDATE_ROOTS = ["../..", ".", "../../.."]
DATA_ROOT = next((r for r in CANDIDATE_ROOTS if os.path.isfile(os.path.join(r, "full_df.csv"))), None)
assert DATA_ROOT is not None, "Could not locate full_df.csv -- run this notebook from Code/notebooks/"
DATA_ROOT = os.path.abspath(DATA_ROOT)
IMG_DIR = os.environ.get("MC_IMG_DIR", os.path.join(DATA_ROOT, "preprocessed_images"))
OUT_DATA = os.path.join(DATA_ROOT, "Data")
OUT_MODELS = os.path.join(DATA_ROOT, "Code", "models")
OUT_FIGS = os.path.join(DATA_ROOT, "Code", "notebooks")

# SMOKE=1 is only used to test that the code runs (tiny subset, small images). Leave it unset for the real run.
SMOKE = os.environ.get("MC_SMOKE") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DATA_ROOT:", DATA_ROOT)
print("Device   :", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only")
if device.type != "cuda" and not SMOKE:
    print("WARNING: no GPU found -- this notebook is meant for the RTX 4060 and will be very slow on CPU.")

CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
CLASS_NAMES = {"N": "Normal", "D": "Diabetes", "G": "Glaucoma", "C": "Cataract",
               "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Others"}
NAMES = [CLASS_NAMES[c] for c in CLASSES]

IMG_SIZE = 224 if SMOKE else 448
BATCH = 8  # safe for an 8 GB laptop GPU at 448x448
EPOCHS = 1 if SMOKE else 25
PATIENCE = 6


# ======================================================================
# 2. Load the Stage 9 patient-level split (identical train / validation / test)
# ======================================================================
split = pd.read_csv(os.path.join(OUT_DATA, "stage9_multiclass_split.csv"))
split["y"] = split["diagnosis"].map({c: i for i, c in enumerate(CLASSES)})
train_df = split[split["split"] == "train"].reset_index(drop=True)
val_df = split[split["split"] == "val"].reset_index(drop=True)
test_df = split[split["split"] == "test"].reset_index(drop=True)

overlap = len(set(train_df.ID) & set(val_df.ID)) + len(set(train_df.ID) & set(test_df.ID)) + len(set(val_df.ID) & set(test_df.ID))
print("Images  train / val / test:", len(train_df), len(val_df), len(test_df))
print("Patients shared between splits:", overlap)
assert overlap == 0
if SMOKE:
    train_df = train_df.groupby("y").head(8).reset_index(drop=True)
    val_df = val_df.groupby("y").head(4).reset_index(drop=True)
    test_df = test_df.groupby("y").head(4).reset_index(drop=True)
    print("SMOKE mode -- tiny subset:", len(train_df), len(val_df), len(test_df))


# ======================================================================
# 3. Data loading and augmentation (448×448)
# ======================================================================
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class FundusDS(Dataset):
    def __init__(self, d, tf):
        self.files, self.y, self.tf = d["filename"].tolist(), d["y"].tolist(), tf
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        return self.tf(Image.open(os.path.join(IMG_DIR, self.files[i])).convert("RGB")), self.y[i]

pin = device.type == "cuda"
# num_workers=0 keeps this working inside Jupyter on Windows
train_dl = DataLoader(FundusDS(train_df, train_tf), batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=pin)
val_dl = DataLoader(FundusDS(val_df, eval_tf), batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)
test_dl = DataLoader(FundusDS(test_df, eval_tf), batch_size=32, shuffle=False, num_workers=0, pin_memory=pin)
print("Batches per epoch:", len(train_dl))


# ======================================================================
# 4. Model — EfficientNet-B0, whole network trained
# ======================================================================
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

try:
    net = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    pretrained = True
except Exception as e:
    if not SMOKE:
        raise
    print("SMOKE mode: ImageNet weights could not be downloaded here, using random weights:", e)
    net = models.efficientnet_b0(weights=None)
    pretrained = False
net.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(net.classifier[1].in_features, len(CLASSES)))
net = net.to(device)
print("ImageNet pre-trained:", pretrained)
print("Trainable parameters:", sum(p.numel() for p in net.parameters() if p.requires_grad))

counts = np.bincount(train_df["y"], minlength=len(CLASSES)).astype(float)
# square-root inverse frequency: helps rare classes without over-boosting them
w = 1.0 / np.sqrt(np.maximum(counts, 1)); w = w / w.mean()
criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device), label_smoothing=0.1)
print("Class weights:", dict(zip(CLASSES, np.round(w, 2))))

optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
steps_total = EPOCHS * len(train_dl)
warmup = len(train_dl)  # 1 epoch warm-up
def lr_lambda(step):
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps_total - warmup)))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# ======================================================================
# 5. Training (best checkpoint kept by validation macro-F1)
# ======================================================================
from sklearn.metrics import f1_score, accuracy_score

def predict(model, dl, tta=True):
    model.eval()
    probs, ys = [], []
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            p = torch.softmax(model(x).float(), 1)
            if tta:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
            probs.append(p.cpu()); ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()

CKPT = os.path.join(OUT_MODELS, "stage10_multiclass_efficientnet_b0.pt")
history, best_f1, bad = [], -1.0, 0
t_start = time.time()
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    net.train()
    total, n = 0.0, 0
    for x, y in train_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss = criterion(net(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(y); n += len(y)
    vp, vy = predict(net, val_dl, tta=False)
    v_f1 = f1_score(vy, vp.argmax(1), average="macro", zero_division=0)
    v_acc = accuracy_score(vy, vp.argmax(1))
    history.append({"epoch": epoch, "train_loss": total / n, "val_acc": v_acc, "val_f1_macro": v_f1})
    better = v_f1 > best_f1
    if better:
        best_f1, bad = v_f1, 0
        torch.save(net.state_dict(), CKPT)
    else:
        bad += 1
    print(f"Epoch {epoch:2d} | loss {total/n:.4f} | val acc {v_acc:.3f} | val macro-F1 {v_f1:.3f}"
          f"{'  <- best, saved' if better else ''} | {time.time()-t0:.0f}s", flush=True)
    if bad >= PATIENCE:
        print(f"Early stopping: no validation improvement for {PATIENCE} epochs")
        break
print(f"Training time: {(time.time()-t_start)/60:.1f} min | best val macro-F1: {best_f1:.3f}")

hist = pd.DataFrame(history)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(hist.epoch, hist.train_loss, marker="o"); ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch")
ax[1].plot(hist.epoch, hist.val_acc, marker="o", label="val accuracy")
ax[1].plot(hist.epoch, hist.val_f1_macro, marker="o", label="val macro-F1")
ax[1].set_title("Validation"); ax[1].set_xlabel("Epoch"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_FIGS, "stage10_training_curves.png"), dpi=150); plt.close()


# ======================================================================
# 6. Test-set results (8 classes), with test-time augmentation
# ======================================================================
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

net.load_state_dict(torch.load(CKPT, map_location=device))
val_p, val_y = predict(net, val_dl, tta=True)
val_f1_tta = f1_score(val_y, val_p.argmax(1), average="macro", zero_division=0)
tp, ty = predict(net, test_dl, tta=True)
pred = tp.argmax(1)

def metrics(y, p):
    return {"accuracy": accuracy_score(y, p),
            "precision_macro": precision_score(y, p, average="macro", zero_division=0),
            "recall_macro": recall_score(y, p, average="macro", zero_division=0),
            "f1_macro": f1_score(y, p, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, p, average="weighted", zero_division=0)}
m10 = metrics(ty, pred)
print("Validation macro-F1 (with TTA):", round(val_f1_tta, 3))
print({k: round(v, 3) for k, v in m10.items()})
print()
print(classification_report(ty, pred, labels=range(len(CLASSES)), target_names=NAMES, zero_division=0))

cm = confusion_matrix(ty, pred, labels=range(len(CLASSES)))
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(NAMES, rotation=35, ha="right")
ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(NAMES)
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Stage 10 (EfficientNet-B0, 448px) — test confusion matrix\n(numbers = images, colour = row %)")
plt.colorbar(im, fraction=0.046); plt.tight_layout()
plt.savefig(os.path.join(OUT_FIGS, "stage10_confusion_matrix.png"), dpi=150); plt.close()


# ======================================================================
# 7. Per-disease yes/no results (one disease vs all others)
# ======================================================================
from sklearn.metrics import roc_auc_score

rows = []
for i, name in enumerate(NAMES):
    yt = (ty == i).astype(int); yp = (pred == i).astype(int)
    tp_ = int(((yt == 1) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    rows.append({"disease": name, "test_images": int(yt.sum()),
                 "auc": roc_auc_score(yt, tp[:, i]) if 0 < yt.sum() < len(yt) else float("nan"),
                 "sensitivity": tp_ / max(tp_ + fn, 1), "specificity": tn / max(tn + fp, 1),
                 "precision": tp_ / max(tp_ + fp, 1), "f1": f1_score(yt, yp, zero_division=0)})
ovr = pd.DataFrame(rows).round(3)
print(ovr.to_string(index=False))
print("\nMean AUC over 8 classes:", round(ovr["auc"].mean(), 3))


# ======================================================================
# 8. Compare with Stage 9 and choose the final 8-class model (by validation, not test)
# ======================================================================
s9 = json.load(open(os.path.join(OUT_DATA, "stage9_multiclass_config.json")))
comp = pd.DataFrame([
    {"model": "Stage 9 — MobileNetV2, 224px, last 5 blocks (CPU)", "val_f1_macro": s9["best_val_f1_macro"], **s9["test_metrics"]},
    {"model": "Stage 10 — EfficientNet-B0, 448px, full fine-tune (GPU)", "val_f1_macro": val_f1_tta, **m10},
]).set_index("model")
print(comp.round(3).to_string())

winner = "stage10" if val_f1_tta > s9["best_val_f1_macro"] else "stage9"
print("\nSelected by validation macro-F1:", winner)

if not SMOKE:
    comp.to_csv(os.path.join(OUT_DATA, "stage10_comparison.csv"))
    ovr.to_csv(os.path.join(OUT_DATA, "stage10_per_disease_yes_no.csv"), index=False)
    per_class = pd.DataFrame({
        "class": NAMES, "test_images": np.bincount(ty, minlength=len(CLASSES)), "train_images": counts.astype(int),
        "cnn_recall": recall_score(ty, pred, average=None, labels=range(len(CLASSES)), zero_division=0),
        "cnn_precision": precision_score(ty, pred, average=None, labels=range(len(CLASSES)), zero_division=0),
        "cnn_f1": f1_score(ty, pred, average=None, labels=range(len(CLASSES)), zero_division=0)}).round(3)
    per_class.to_csv(os.path.join(OUT_DATA, "stage10_per_class.csv"), index=False)
    cfg10 = {"classes": CLASSES, "class_names": CLASS_NAMES, "architecture": "efficientnet_b0", "img_size": IMG_SIZE,
             "checkpoint": "Code/models/stage10_multiclass_efficientnet_b0.pt", "tta": True,
             "best_val_f1_macro": float(val_f1_tta), "epochs_run": len(history), "device_used": str(device),
             "per_class_csv": "Data/stage10_per_class.csv",
             "test_metrics": {k: float(v) for k, v in m10.items()}}
    json.dump(cfg10, open(os.path.join(OUT_DATA, "stage10_multiclass_config.json"), "w"), indent=2)
    print("Saved: Code/models/stage10_multiclass_efficientnet_b0.pt, Data/stage10_comparison.csv,",
          "Data/stage10_per_class.csv, Data/stage10_per_disease_yes_no.csv, Data/stage10_multiclass_config.json")
else:
    print("SMOKE mode: nothing saved except the throwaway checkpoint.")
print("\nDONE -- Stage 10 finished.")


import matplotlib.pyplot as _plt
_plt.close("all")
sys.stdout = sys.__stdout__
_log.close()
