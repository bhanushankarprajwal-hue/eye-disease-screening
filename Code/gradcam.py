"""
Grad-CAM heatmaps: show WHERE in the fundus image a model looked to make its decision.

How it works (Selvaraju et al., 2017):
  1. Run the image through the network and keep the feature maps of the last convolution block.
  2. Back-propagate the score of the class we care about to those feature maps.
  3. Weight each feature map by how much it pushed that score up, add them, keep only positive values.
  4. Resize to the image and colour it: red/yellow = most important, blue = least.

No extra libraries -- just PyTorch hooks. Used by Code/app.py.
"""

import numpy as np
import torch
from matplotlib import cm
from PIL import Image


def _cam(model_forward, target_layer, x, score_fn):
    """Generic Grad-CAM. model_forward(x) -> output; score_fn(output) -> scalar to explain."""
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = target_layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = model_forward(x)
            score = score_fn(out)
            score.backward()
    finally:
        h.remove()
    act, grad = store["act"][0], store["grad"][0]           # (C, H, W)
    weights = grad.mean(dim=(1, 2))                          # one weight per feature map
    cam = torch.relu((weights[:, None, None] * act).sum(0))  # (H, W)
    cam = cam.detach().cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def overlay(rgb_img, cam, alpha=0.45):
    """Puts the heatmap on top of the image. rgb_img: PIL image; cam: 2-D array in [0, 1]."""
    w, h = rgb_img.size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((w, h), Image.BILINEAR)
    heat = cm.jet(np.asarray(cam_img) / 255.0)[..., :3]
    base = np.asarray(rgb_img.convert("RGB")) / 255.0
    mixed = (1 - alpha) * base + alpha * heat
    return Image.fromarray(np.uint8(np.clip(mixed, 0, 1) * 255))


def multiclass_cam(config, model, device, image_path, class_name):
    """Heatmap for one of the 8 classes, from the 8-disease model (Code/predict_multiclass.py)."""
    import predict_multiclass as pmc
    names = [config["class_names"][c] for c in config["classes"]]
    idx = names.index(class_name)
    img = pmc.load_rgb(image_path)
    x = config["_transform"](img).unsqueeze(0).to(device)
    arch = config.get("architecture", "mobilenet_v2")
    target = model.features[-1]  # last convolution block (EfficientNet-B0 and MobileNetV2 both)
    cam = _cam(model, target, x, lambda out: out[0, idx])
    size = config.get("img_size", 224)
    return overlay(img.resize((size, size)), cam), cam


def myopia_cam(config, bundle, image_path, age, sex):
    """Heatmap for the Myopia score of the Myopia-check model (Code/predict_myopia.py)."""
    import predict_myopia as pm
    device = bundle["device"]
    img = Image.open(image_path).convert("RGB")
    a = np.asarray(img)
    if a[..., 2].mean() > a[..., 0].mean() + 10:  # same blue-image colour fix as the predictor
        img = Image.fromarray(np.ascontiguousarray(a[..., ::-1]))
    x = pm.eval_transform(img).unsqueeze(0).to(device)
    clin = torch.tensor([[pm.scale_age(age, config), pm.encode_sex(sex)]], dtype=torch.float32).to(device)
    if bundle["type"] == "finetuned_fusion":
        model = bundle["model"]
        cam = _cam(lambda t: model(t, clin), model.backbone[-1], x, lambda out: out.squeeze())
    else:  # frozen-backbone fusion
        base, head, fusion = bundle["base_model"], bundle["feature_head"], bundle["fusion_model"]

        def fwd(t):
            f = base.features(t)
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
            return fusion(head(f), clin)
        cam = _cam(fwd, base.features[-1], x, lambda out: out.squeeze())
    return overlay(img.resize((224, 224)), cam), cam
