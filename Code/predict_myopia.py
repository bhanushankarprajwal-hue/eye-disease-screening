"""
Stage 8 - Final inference script.

Loads whichever model Stage 8 (Code/notebooks/08_final_model.ipynb) selected as the
final model -- reading Data/stage8_final_model_config.json rather than hardcoding an
architecture, so this script keeps working whichever of the Stage 8 experiments (or
the original Stage 7 model) ended up winning.

Usage (from the project root, with venv311 active):

    python Code/predict_myopia.py --age 34 --sex Male --image1 preprocessed_images/0_left.jpg
    python Code/predict_myopia.py --age 60 --sex Female --image1 preprocessed_images/12_left.jpg --image2 preprocessed_images/12_right.jpg

Both --image2 and --sex are optional; if only one eye's image is available, pass just --image1.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "Data", "stage8_final_model_config.json")

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_stage6_architecture():
    m = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )
    return m


class FusionModel(nn.Module):
    """Used when the final model is a frozen-backbone fusion model (Stage 7 or Stage 8 Experiment A)."""

    def __init__(self, clinical_dim=2, image_dim=64, clinical_hidden=8, fusion_hidden=32, dropout=0.3):
        super().__init__()
        self.clinical_branch = nn.Sequential(nn.Linear(clinical_dim, clinical_hidden), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + clinical_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, image_feat, clinical_feat):
        clin = self.clinical_branch(clinical_feat)
        combined = torch.cat([image_feat, clin], dim=1)
        return self.classifier(combined)


class FineTunedFusionModel(nn.Module):
    """Used when the final model is Stage 8 Experiment B (fine-tuned MobileNetV2 backbone)."""

    def __init__(self, backbone_ckpt_path, unfreeze_from_block, clinical_dim=2,
                 clinical_hidden=8, fusion_hidden=32, dropout=0.3, device="cpu"):
        super().__init__()
        base = build_stage6_architecture()
        base.load_state_dict(torch.load(backbone_ckpt_path, map_location=device))
        self.backbone = base.features
        for i, block in enumerate(self.backbone):
            for p in block.parameters():
                p.requires_grad = (i >= unfreeze_from_block)
        self.image_head = nn.Sequential(*list(base.classifier.children())[:3])

        self.clinical_branch = nn.Sequential(nn.Linear(clinical_dim, clinical_hidden), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(64 + clinical_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, images, clinical_feat):
        feats = self.backbone(images)
        feats = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
        img_feat = self.image_head(feats)
        clin = self.clinical_branch(clinical_feat)
        combined = torch.cat([img_feat, clin], dim=1)
        return self.classifier(combined)


def load_config():
    if not os.path.isfile(CONFIG_PATH):
        raise FileNotFoundError(
            f"Missing {CONFIG_PATH}. Run Code/notebooks/08_final_model.ipynb first -- "
            "it writes this config once the final model is selected."
        )
    with open(CONFIG_PATH) as f:
        return json.load(f)


def encode_sex(sex_str):
    if sex_str is None:
        return 0.0  # unknown sex -- falls back to the more common encoded value used in training
    return 1.0 if str(sex_str).strip().lower().startswith("m") else 0.0


def scale_age(age, config):
    scaler = config["clinical_scaler"]
    age = float(age) if age is not None else scaler["age_median"]
    return (age - scaler["age_mean"]) / scaler["age_std"]


def load_image_tensor(path, device):
    img = Image.open(path).convert("RGB")
    a = np.asarray(img)
    if a[..., 2].mean() > a[..., 0].mean() + 10:  # blue-looking fundus = BGR-saved image; swap back to RGB
        img = Image.fromarray(np.ascontiguousarray(a[..., ::-1]))
    tensor = eval_transform(img).unsqueeze(0).to(device)
    return tensor


def load_final_model(device=None):
    """Loads the final Stage 8 model ONCE (config + weights) so repeated predictions don't
    reload the checkpoint from disk every time -- used by the UI (app.py) to cache the model
    for the life of the app session. Returns (config, bundle)."""
    config = load_config()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["final_model"] == "frozen_fusion":
        backbone_ckpt = os.path.join(PROJECT_ROOT, config["backbone_checkpoint"])
        fusion_ckpt = os.path.join(PROJECT_ROOT, config["fusion_checkpoint"])

        base_model = build_stage6_architecture()
        base_model.load_state_dict(torch.load(backbone_ckpt, map_location=device))
        base_model = base_model.to(device)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False
        feature_head = nn.Sequential(*list(base_model.classifier.children())[:3]).to(device)

        fusion_cfg = config.get("fusion_config", {})
        dropout = fusion_cfg.get("dropout", 0.3) if isinstance(fusion_cfg.get("dropout", 0.3), float) else 0.3
        fusion_model = FusionModel(dropout=dropout).to(device)
        fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        fusion_model.eval()

        bundle = {"type": "frozen_fusion", "base_model": base_model,
                  "feature_head": feature_head, "fusion_model": fusion_model, "device": device}

    elif config["final_model"] == "finetuned_fusion":
        backbone_ckpt = os.path.join(PROJECT_ROOT, "Code", "models",
                                      "stage6_myopia_vs_normal_mobilenetv2_local_gpu.pt")
        fusion_ckpt = os.path.join(PROJECT_ROOT, config["fusion_checkpoint"])

        model = FineTunedFusionModel(
            backbone_ckpt_path=backbone_ckpt,
            unfreeze_from_block=config["unfreeze_from_block"],
            device=device,
        ).to(device)
        model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        model.eval()

        bundle = {"type": "finetuned_fusion", "model": model, "device": device}

    else:
        raise ValueError(f"Unknown final_model type in config: {config['final_model']}")

    return config, bundle


def predict_with_loaded(bundle, config, age, sex, image_path1, image_path2=None):
    """Runs one prediction using a model already loaded by load_final_model() -- no disk I/O
    for weights here, just image loading + a forward pass. Returns (probability, predicted_label,
    threshold_used, model_description)."""
    device = bundle["device"]
    clin = torch.tensor(
        [[scale_age(age, config), encode_sex(sex)]], dtype=torch.float32
    ).to(device)

    image_paths = [p for p in [image_path1, image_path2] if p]
    if not image_paths:
        raise ValueError("At least one fundus image path is required.")
    for p in image_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Image not found: {p}")

    if bundle["type"] == "frozen_fusion":
        base_model = bundle["base_model"]
        feature_head = bundle["feature_head"]
        fusion_model = bundle["fusion_model"]

        image_feats = []
        with torch.no_grad():
            for p in image_paths:
                img_t = load_image_tensor(p, device)
                feats = base_model.features(img_t)
                feats = nn.functional.adaptive_avg_pool2d(feats, 1).flatten(1)
                feats = feature_head(feats)
                image_feats.append(feats)
            image_feat = torch.stack(image_feats, dim=0).mean(dim=0)
            logit = fusion_model(image_feat, clin).squeeze()
            prob = torch.sigmoid(logit).item()

    elif bundle["type"] == "finetuned_fusion":
        model = bundle["model"]
        probs = []
        with torch.no_grad():
            for p in image_paths:
                img_t = load_image_tensor(p, device)
                logit = model(img_t, clin).squeeze()
                probs.append(torch.sigmoid(logit).item())
        prob = float(np.mean(probs))

    else:
        raise ValueError(f"Unknown bundle type: {bundle['type']}")

    threshold = config.get("threshold", 0.5)
    label = "Myopia" if prob > threshold else "Non-Myopia"
    return prob, label, threshold, config.get("description", config["final_model"])


def predict_myopia(age, sex, image_path1, image_path2=None, device=None):
    """Returns (probability, predicted_label, threshold_used, model_description).

    Convenience one-shot wrapper (loads the model fresh every call) -- used by the CLI below
    and by anything that only needs a single prediction. For repeated predictions (e.g. the
    Streamlit UI in app.py), call load_final_model() once and reuse it with predict_with_loaded()
    instead, which skips reloading the checkpoint from disk each time."""
    config, bundle = load_final_model(device=device)
    return predict_with_loaded(bundle, config, age, sex, image_path1, image_path2)


def main():
    parser = argparse.ArgumentParser(description="Predict Myopia vs Non-Myopia from clinical + fundus image data.")
    parser.add_argument("--age", type=float, default=None, help="Patient age (years). Missing -> imputed with the training median.")
    parser.add_argument("--sex", type=str, default=None, help="Patient sex ('Male' or 'Female'). Missing -> falls back to the encoded default.")
    parser.add_argument("--image1", type=str, required=True, help="Path to a fundus image (left or right eye).")
    parser.add_argument("--image2", type=str, default=None, help="Optional second fundus image (the other eye).")
    args = parser.parse_args()

    prob, label, threshold, description = predict_myopia(args.age, args.sex, args.image1, args.image2)

    print("Final model:", description)
    print(f"Decision threshold: {threshold:.2f}")
    print(f"Predicted Myopia probability: {prob:.4f}")
    print(f"Predicted label: {label}")


if __name__ == "__main__":
    main()
