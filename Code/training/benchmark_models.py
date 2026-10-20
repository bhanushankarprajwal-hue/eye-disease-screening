"""
Benchmark: our model vs other well-known CNN / Transformer models, trained and tested in exactly the same way.

Why: to answer "did you compare your model with other models?" fairly. Every model below gets
  * the same images and the same patient-level split (Data/stage9_multiclass_split.csv: 4,472 train /
    960 validation / 960 test photos, no patient in two parts),
  * the same image size (384 px), the same augmentation, loss (class-weighted, label smoothing), optimiser,
    learning-rate schedule, number of epochs and early stopping,
  * ImageNet pre-trained weights (transfer learning),
  * the best epoch chosen on the VALIDATION set only, then ONE evaluation on the test set (with mirror TTA).
Only the network changes, so the differences come from the architecture.

Models: MobileNetV2, EfficientNet-B0, ResNet-50, DenseNet-121, VGG-16, ConvNeXt-Tiny, ViT-B/16 (at 224 px,
its native size).
Then a LEARNING CURVE: EfficientNet-B0 retrained on 25%, 50% and 75% of the training patients (100% = the run
above), to show how much accuracy depends on the amount of training data. Turn off with LEARNING_CURVE=0. Results are written after each model, so the script can be stopped and started again:
models that are already finished are skipped.

Run from the project root with venv311 active (GPU strongly recommended, about 3-4 hours in total):
    python Code\\benchmark_models.py
Only some models:      set BENCH_MODELS=resnet50,densenet121   (then run the command above)
Outputs: Data/benchmark_results.csv, Data/learning_curve_results.csv, Data/benchmark_log.txt,
         Code/notebooks/benchmark_chart.png, Code/notebooks/learning_curve.png
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE = os.environ.get("MC_SMOKE") == "1"          # quick code test on a tiny subset; leave unset for the real run
IMG_DIR = os.environ.get("MC_IMG_DIR", os.path.join(ROOT, "preprocessed_images"))
RESULTS = os.path.join(ROOT, "Data", "benchmark_results.csv" if not SMOKE else "benchmark_results_SMOKE.csv")
LC_RESULTS = os.path.join(ROOT, "Data", "learning_curve_results.csv" if not SMOKE else "learning_curve_SMOKE.csv")
CLASSES = ["N", "D", "G", "C", "A", "H", "M", "O"]
NAMES = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Others"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG = 96 if SMOKE else 384
EPOCHS = 1 if SMOKE else 12
PATIENCE = 4
WARMUP = 0 if SMOKE else 1
SEED = 42

# name: (builder, weights enum name, input size, batch size, learning rate)
MODELS = {
    "mobilenet_v2":    ("mobilenet_v2", "MobileNet_V2_Weights", IMG, 16, 3e-4),
    "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights", IMG, 16, 3e-4),
    "resnet50":        ("resnet50", "ResNet50_Weights", IMG, 16, 3e-4),
    "densenet121":     ("densenet121", "DenseNet121_Weights", IMG, 12, 3e-4),
    "vgg16":           ("vgg16_bn", "VGG16_BN_Weights", IMG, 8, 1e-4),
    "convnext_tiny":   ("convnext_tiny", "ConvNeXt_Tiny_Weights", IMG, 12, 1e-4),
    "vit_b_16":        ("vit_b_16", "ViT_B_16_Weights", 224, 16, 5e-5),   # ViT-B/16 only accepts 224 px
}
PRETTY = {"mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50",
          "densenet121": "DenseNet-121", "vgg16": "VGG-16 (BN)", "convnext_tiny": "ConvNeXt-Tiny",
          "vit_b_16": "ViT-B/16 (Transformer)"}


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, t):
        for x in self.s: x.write(t); x.flush()
    def flush(self):
        for x in self.s: x.flush()


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    if a[..., 2].mean() > a[..., 0].mean() + 10:      # BGR-saved photo -> swap back (same rule as the app)
        img = Image.fromarray(np.ascontiguousarray(a[..., ::-1]))
    return img


class Cached(Dataset):
    """Images are decoded and resized ONCE and kept in memory (much faster epochs)."""
    def __init__(self, arrays, labels, tf):
        self.a, self.y, self.tf = arrays, labels, tf
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.a[i])), self.y[i]


def cache(df, size):
    out = np.zeros((len(df), size, size, 3), dtype=np.uint8)
    for i, f in enumerate(df.filename):
        out[i] = np.asarray(load_rgb(os.path.join(IMG_DIR, f)).resize((size, size), Image.BILINEAR))
    return out


def build(key, n_classes, device):
    fn, wname, *_ = MODELS[key]
    pretrained = True
    try:
        weights = getattr(models, wname).IMAGENET1K_V1
        m = getattr(models, fn)(weights=weights)
    except Exception as e:  # noqa: BLE001
        if not SMOKE:
            raise
        print(f"   (SMOKE: ImageNet weights not available here -> random weights: {e.__class__.__name__})")
        m = getattr(models, fn)(weights=None); pretrained = False
    # replace the last layer with an 8-class head
    if key in ("mobilenet_v2", "efficientnet_b0"):
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif key == "resnet50":
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif key == "densenet121":
        m.classifier = nn.Linear(m.classifier.in_features, n_classes)
    elif key == "vgg16":
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif key == "convnext_tiny":
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    elif key == "vit_b_16":
        m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
    return m.to(device), pretrained


def main():
    os.chdir(ROOT)
    log = open(os.path.join(ROOT, "Data", "benchmark_log.txt"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log)
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print(time.strftime("%Y-%m-%d %H:%M"), "| device:", device,
          "|", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU only")
    if device.type != "cuda" and not SMOKE:
        print("WARNING: no GPU found -- this will be extremely slow on CPU.")

    sp = pd.read_csv(os.path.join("Data", "stage9_multiclass_split.csv"))
    sp["y"] = sp.diagnosis.map({c: i for i, c in enumerate(CLASSES)})
    tr, va, te = (sp[sp.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    assert not (set(tr.ID) & set(te.ID)) and not (set(tr.ID) & set(va.ID)) and not (set(va.ID) & set(te.ID))
    if SMOKE:
        tr, va, te = (d.groupby("y").head(n).reset_index(drop=True) for d, n in ((tr, 3), (va, 2), (te, 2)))
        for d in (tr, va, te):
            d.drop(d[~d.filename.apply(lambda f: os.path.isfile(os.path.join(IMG_DIR, f)))].index, inplace=True)
    print(f"Images train/val/test: {len(tr)}/{len(va)}/{len(te)} (patient-level split, no overlap)")

    todo = [k for k in MODELS if k in os.environ.get("BENCH_MODELS", ",".join(MODELS)).split(",")]
    done = pd.read_csv(RESULTS) if os.path.isfile(RESULTS) else pd.DataFrame(columns=["key"])
    todo = [k for k in todo if k not in set(done.key)]
    print("Models to train:", ", ".join(PRETTY[k] for k in todo) or "none (all finished)")

    caches = {}

    def get_cache(size):
        if size not in caches:
            t0 = time.time()
            caches[size] = tuple(cache(d, size) for d in (tr, va, te))
            print(f"Cached {len(tr)+len(va)+len(te)} images at {size}px in {time.time()-t0:.0f}s")
        return caches[size]

    use_amp = device.type == "cuda"

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

    def train_one(key, trd, a_tr, tag=""):
        """Trains one network on the given training rows and returns its result row (test set is always the same)."""
        fn, wname, size, bs, lr = MODELS[key]
        print(f"\n----- {PRETTY[key]}{(' ' + tag) if tag else ''}  ({size}px, {len(trd)} training photos, batch {bs}, lr {lr}) -----")
        _a, a_va, a_te = get_cache(size)
        cw = np.bincount(trd.y, minlength=8).astype(float)
        cw = 1.0 / np.sqrt(np.maximum(cw, 1)); cw = cw / cw.mean()
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        g = torch.Generator(); g.manual_seed(SEED)
        train_dl = DataLoader(Cached(a_tr, trd.y.tolist(), train_tf), batch_size=bs, shuffle=True, generator=g,
                              num_workers=0, pin_memory=use_amp)
        val_dl = DataLoader(Cached(a_va, va.y.tolist(), eval_tf), batch_size=32, num_workers=0)
        test_dl = DataLoader(Cached(a_te, te.y.tolist(), eval_tf), batch_size=32, num_workers=0)

        torch.manual_seed(SEED)
        net, pretrained = build(key, 8, device)
        n_params = sum(p.numel() for p in net.parameters()) / 1e6
        crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=device), label_smoothing=0.1)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        total = EPOCHS * len(train_dl); warm = max(1, WARMUP * len(train_dl))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ckpt = os.path.join("Code", "models", f"benchmark_{key}{tag.replace(' ', '_')}{'_SMOKE' if SMOKE else ''}.pt")
        best, bad, best_ep, t0 = -1.0, 0, 0, time.time()
        for ep in range(1, EPOCHS + 1):
            e0 = time.time(); net.train(); tot = n = 0
            for x, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss = crit(net(x), y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item() * len(y); n += len(y)
            vp = predict(net, val_dl, tta=False)
            vf1 = f1_score(va.y, vp.argmax(1), average="macro", zero_division=0)
            mark = ""
            if vf1 > best:
                best, bad, best_ep, mark = vf1, 0, ep, "  <- best, saved"
                torch.save(net.state_dict(), ckpt)
            else:
                bad += 1
            print(f"  epoch {ep:2d} | loss {tot/n:.4f} | val acc {accuracy_score(va.y, vp.argmax(1)):.3f} "
                  f"| val macro-F1 {vf1:.3f}{mark} | {time.time()-e0:.0f}s", flush=True)
            if bad >= PATIENCE:
                print(f"  early stop (no improvement for {PATIENCE} epochs)"); break
        train_min = (time.time() - t0) / 60

        net.load_state_dict(torch.load(ckpt, map_location=device))
        vp = predict(net, val_dl); tp = predict(net, test_dl)
        # speed: time for one photo (batch of 1, no TTA), averaged
        x1 = torch.randn(1, 3, size, size, device=device)
        net.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for _ in range(3): net(x1)
            if device.type == "cuda": torch.cuda.synchronize()
            s0 = time.time()
            for _ in range(20): net(x1)
            if device.type == "cuda": torch.cuda.synchronize()
        ms = (time.time() - s0) / 20 * 1000
        y, pred = te.y.values, tp.argmax(1)
        aucs = [roc_auc_score((y == i).astype(int), tp[:, i]) for i in range(8) if 0 < (y == i).sum() < len(y)]
        row = {"key": key, "model": PRETTY[key], "img_size": size, "params_M": round(n_params, 1),
               "pretrained": pretrained, "epochs_run": ep, "best_epoch": best_ep, "train_minutes": round(train_min, 1),
               "ms_per_image": round(ms, 1),
               "val_f1_macro": f1_score(va.y, vp.argmax(1), average="macro", zero_division=0),
               "test_accuracy": accuracy_score(y, pred),
               "test_f1_macro": f1_score(y, pred, average="macro", zero_division=0),
               "test_f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
               "test_precision_macro": precision_score(y, pred, average="macro", zero_division=0),
               "test_recall_macro": recall_score(y, pred, average="macro", zero_division=0),
               "test_mean_auc": float(np.mean(aucs))}
        for i, nm in enumerate(NAMES):
            row[f"f1_{nm}"] = f1_score((y == i).astype(int), (pred == i).astype(int), zero_division=0)
        print(f"  TEST: accuracy {row['test_accuracy']:.3f} | macro-F1 {row['test_f1_macro']:.3f} | "
              f"mean AUC {row['test_mean_auc']:.3f} | {n_params:.1f}M params | {ms:.1f} ms/photo | {train_min:.0f} min")
        row["train_photos"] = len(trd)
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return row

    for key in todo:
        a_tr = get_cache(MODELS[key][2])[0]
        row = train_one(key, tr, a_tr)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(RESULTS, index=False)

    # ---------------------------------------------------------------- learning curve (EfficientNet-B0)
    # Same network and recipe, trained on 25% / 50% / 75% of the training PATIENTS (100% = the benchmark run above).
    # Shows whether accuracy is still rising with more data. Test photos are always the same 960.
    if os.environ.get("LEARNING_CURVE", "1") == "1" and "efficientnet_b0" in set(done.key):
        lc = pd.read_csv(LC_RESULTS) if os.path.isfile(LC_RESULTS) else pd.DataFrame(columns=["fraction"])
        full = done[done.key == "efficientnet_b0"].iloc[0].to_dict()
        if 1.0 not in set(lc.fraction):
            lc = pd.concat([lc, pd.DataFrame([{**full, "fraction": 1.0}])], ignore_index=True)
        pats = tr.groupby("ID").y.first().reset_index()           # one label per patient, for stratified sampling
        for frac in (0.25, 0.5, 0.75):
            if frac in set(lc.fraction):
                continue
            rng = np.random.RandomState(SEED)
            keep = []
            for _cls, grp in pats.groupby("y"):                    # stratified by class, sampled by PATIENT
                k = max(1, int(round(frac * len(grp))))
                keep += list(rng.choice(grp.ID.values, size=k, replace=False))
            mask = tr.ID.isin(set(keep)).values
            trd = tr[mask].reset_index(drop=True)
            a_full = get_cache(MODELS["efficientnet_b0"][2])[0]
            row = train_one("efficientnet_b0", trd, a_full[mask], tag=f"{int(frac*100)}pct")
            row["fraction"] = frac
            lc = pd.concat([lc, pd.DataFrame([row])], ignore_index=True)
            lc.to_csv(LC_RESULTS, index=False)
        lc = lc.sort_values("fraction")
        lc.to_csv(LC_RESULTS, index=False)
        print("\nLEARNING CURVE (EfficientNet-B0, same 960 test photos):")
        print(lc[["fraction", "train_photos", "test_accuracy", "test_f1_macro", "test_mean_auc"]].round(3).to_string(index=False))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs = lc.train_photos.astype(float)
        ax.plot(xs, lc.test_accuracy * 100, "o-", label="Accuracy", color="#2F6BFF")
        ax.plot(xs, lc.test_f1_macro * 100, "s-", label="Macro-F1", color="#0E9F9A")
        ax.plot(xs, lc.test_mean_auc * 100, "^-", label="Mean AUC (x100)", color="#F59E0B")
        for x, a in zip(xs, lc.test_accuracy * 100):
            ax.annotate(f"{a:.1f}%", (x, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_xlabel("Number of training photos"); ax.set_ylabel("Score on 960 unseen test photos (%)")
        ax.set_title("Learning curve - EfficientNet-B0 trained on 25/50/75/100% of the training patients")
        ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout()
        if not SMOKE:
            plt.savefig(os.path.join("Code", "notebooks", "learning_curve.png"), dpi=150)
        plt.close()

    # ---------------------------------------------------------------- our final model on the same test photos
    if not done.empty and "ours_final" not in set(done.key):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import predict_multiclass as pmc
        import triage as tr_mod
        cfg, _m, dev = pmc.load_multiclass_model()
        paths = [os.path.join(IMG_DIR, f) for f in te.filename]
        P, _err = tr_mod.predict_many(cfg, dev, paths, batch_size=16)
        y, pred = te.y.values, P.argmax(1)
        aucs = [roc_auc_score((y == i).astype(int), P[:, i]) for i in range(8) if 0 < (y == i).sum() < len(y)]
        n_params = sum(sum(p.numel() for p in mb["model"].parameters()) for mb in cfg["_members"]) / 1e6
        row = {"key": "ours_final", "model": f"OUR FINAL: {pmc.describe(cfg)}", "img_size": "448 + 512",
               "params_M": round(n_params, 1), "pretrained": True, "epochs_run": "", "best_epoch": "",
               "train_minutes": "", "ms_per_image": "",
               "val_f1_macro": float(cfg.get("combined_val_f1_macro", cfg.get("best_val_f1_macro", np.nan))),
               "test_accuracy": accuracy_score(y, pred),
               "test_f1_macro": f1_score(y, pred, average="macro", zero_division=0),
               "test_f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
               "test_precision_macro": precision_score(y, pred, average="macro", zero_division=0),
               "test_recall_macro": recall_score(y, pred, average="macro", zero_division=0),
               "test_mean_auc": float(np.mean(aucs))}
        for i, nm in enumerate(NAMES):
            row[f"f1_{nm}"] = f1_score((y == i).astype(int), (pred == i).astype(int), zero_division=0)
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
        done.to_csv(RESULTS, index=False)

    # ---------------------------------------------------------------- summary table + chart
    show = done.copy()
    show = show.sort_values("test_f1_macro", ascending=False)
    pd.set_option("display.width", 220)
    print("\nRESULTS on the same 960 test photos (sorted by macro-F1):")
    cols = ["model", "img_size", "params_M", "test_accuracy", "test_f1_macro", "test_mean_auc", "ms_per_image", "train_minutes"]
    print(show[cols].round(3).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = show.iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.55 * len(s) + 1.8))
    colors = ["#0E9F9A" if k == "ours_final" else ("#2F6BFF" if k == "efficientnet_b0" else "#94A3B8") for k in s.key]
    for a, col, title in ((ax[0], "test_accuracy", "Test accuracy"), (ax[1], "test_f1_macro", "Test macro-F1")):
        a.barh(s.model, s[col] * 100, color=colors)
        for yy, v in enumerate(s[col] * 100):
            a.text(v + 0.5, yy, f"{v:.1f}%", va="center", fontsize=9)
        a.set_title(title); a.set_xlim(0, 100); a.grid(axis="x", alpha=0.3)
    ax[1].set_yticklabels([])
    fig.suptitle("Same data, same split, same training recipe - 960 unseen ODIR test photos", fontsize=11)
    plt.tight_layout()
    if not SMOKE:
        plt.savefig(os.path.join("Code", "notebooks", "benchmark_chart.png"), dpi=150)
    plt.close()
    print("\nSaved:", RESULTS, "and Code/notebooks/benchmark_chart.png")
    print("DONE -- benchmark finished.")


if __name__ == "__main__":
    main()
