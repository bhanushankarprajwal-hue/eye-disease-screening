"""
8-disease inference (image only).

Uses whichever 8-class model scored best on the VALIDATION set:
  - Stage 9  : MobileNetV2, 224px                        (Data/stage9_multiclass_config.json)
  - Stage 10 : EfficientNet-B0, 448px                    (Data/stage10_multiclass_config.json)
  - Stage 12 : ensemble of Stage 10 + EfficientNet-B0 at 512px with contrast enhancement
                                                          (Data/stage12_multiclass_config.json)
Classes: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Others.

Age/sex are NOT used: Stage 9 showed they carry almost no signal for the 8-class task.

Usage (from the project root, with venv311 active):
    python Code/predict_multiclass.py --image1 test_images/07_Cataract_999_right.jpg
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torchvision import models, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "Data", "stage12_multiclass_config.json"),
    os.path.join(PROJECT_ROOT, "Data", "stage10_multiclass_config.json"),
    os.path.join(PROJECT_ROOT, "Data", "stage9_multiclass_config.json"),
]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def pick_config():
    """Returns the config of the selected model. If Data/multiclass_final_choice.json exists (written by
    Code/train_stage13.py after comparing all models on ODIR + RFMiD validation images) it decides; otherwise the
    config with the best ODIR validation macro-F1 is used (never chosen by test score)."""
    choice = os.path.join(PROJECT_ROOT, "Data", "multiclass_final_choice.json")
    if os.path.isfile(choice):
        with open(choice) as f:
            path = os.path.join(PROJECT_ROOT, json.load(f)["config"])
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    found = []
    for p in CONFIG_CANDIDATES:
        if os.path.isfile(p):
            with open(p) as f:
                found.append(json.load(f))
    if not found:
        raise FileNotFoundError("No 8-class model config found. Run Code/notebooks/09_multiclass.ipynb "
                                "(or Code/train_stage10.py) first.")
    return max(found, key=lambda c: c.get("best_val_f1_macro", 0))


def build_model(arch, n):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.classifier[1].in_features, n))
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, n))
    elif arch == "densenet121":            # trained by Code/benchmark_models.py
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, n)
    elif arch == "convnext_tiny":          # trained by Code/benchmark_models.py
        m = models.convnext_tiny(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n)
    else:
        raise ValueError(f"Unknown architecture in config: {arch}")
    return m  # architecture only; trained weights come from the checkpoint


def enhance(img, size):
    """Ben Graham local contrast enhancement -- identical to Code/train_stage12.py."""
    img = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(img).astype(np.float32)
    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=size / 30))).astype(np.float32)
    out = np.clip(4 * a - 4 * blur + 128, 0, 255)
    yy, xx = np.ogrid[:size, :size]
    out[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 > (0.95 * size / 2) ** 2] = 128
    return Image.fromarray(out.astype(np.uint8))


def _make_transform(size, preprocess):
    if preprocess == "ben_graham":
        return lambda img: _to_tensor(enhance(img, size))
    return lambda img: _to_tensor(img.resize((size, size), Image.BILINEAR))


def load_multiclass_model(device=None):
    """Loads the best 8-class model (or ensemble) once. Returns (config, model, device).
    `model` is the first network (used for the Grad-CAM heatmap); all networks are in config["_members"]."""
    config = pick_config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    member_cfgs = config.get("members") or [config]  # single-model configs describe one network
    members = []
    for mc in member_cfgs:
        m = build_model(mc.get("architecture", "mobilenet_v2"), len(config["classes"]))
        m.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, mc["checkpoint"]), map_location=device))
        m.to(device).eval()
        size = mc.get("img_size", 224)
        members.append({"model": m, "transform": _make_transform(size, mc.get("preprocess", "none")),
                        "tta": mc.get("tta", False), "img_size": size})
    config["_members"] = members
    config["_transform"] = members[0]["transform"]   # used by Code/gradcam.py
    config["img_size"] = members[0]["img_size"]
    return config, members[0]["model"], device


def load_rgb(path):
    """Opens an image as RGB. Fundus photos are always red/orange; if an image is clearly blue
    (more blue than red on average), its colour channels were saved in the wrong order (BGR, a common
    OpenCV mistake -- e.g. the datasets/ and IMAGES/ folders), so they are swapped back."""
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    if a[..., 2].mean() > a[..., 0].mean() + 10:
        img = Image.fromarray(np.ascontiguousarray(a[..., ::-1]))
    return img


def predict_multiclass(config, model, device, image_paths):
    """Returns a list of (class_name, probability), highest first. Averaged over the images, over the
    networks in an ensemble, and over a mirrored copy of each image when the model uses test-time augmentation."""
    image_paths = [p for p in image_paths if p]
    if not image_paths:
        raise ValueError("At least one fundus image path is required.")
    members = config.get("_members") or [{"model": model, "transform": config["_transform"],
                                          "tta": config.get("tta", False)}]
    probs = []
    with torch.no_grad():
        for p in image_paths:
            img = load_rgb(p)
            per_member = []
            for mb in members:
                x = mb["transform"](img).unsqueeze(0).to(device)
                pr = torch.softmax(mb["model"](x), 1)[0]
                if mb["tta"]:
                    pr = (pr + torch.softmax(mb["model"](torch.flip(x, dims=[3])), 1)[0]) / 2
                per_member.append(pr.cpu().numpy())
            probs.append(np.mean(per_member, axis=0))
    mean = np.mean(probs, axis=0)
    names = [config["class_names"][c] for c in config["classes"]]
    return sorted(zip(names, mean.tolist()), key=lambda t: t[1], reverse=True)


def describe(config):
    if config.get("name"):
        return config["name"]
    if config.get("members") and len(config["members"]) > 1:
        return "Stage 12 ensemble: EfficientNet-B0 512px + contrast enhancement, and EfficientNet-B0 448px"
    arch = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2"}.get(config.get("architecture"), "?")
    stage = "Stage 12" if "stage12" in config["checkpoint"] else ("Stage 10" if "stage10" in config["checkpoint"] else "Stage 9")
    return f"{stage} {arch}, {config.get('img_size', 224)}px"


def main():
    ap = argparse.ArgumentParser(description="Predict one of 8 eye-disease classes from fundus image(s).")
    ap.add_argument("--image1", required=True)
    ap.add_argument("--image2", default=None)
    a = ap.parse_args()
    config, model, device = load_multiclass_model()
    print("Model:", describe(config))
    for name, prob in predict_multiclass(config, model, device, [a.image1, a.image2]):
        print(f"{name:13s} {prob*100:5.1f}%")


if __name__ == "__main__":
    main()
