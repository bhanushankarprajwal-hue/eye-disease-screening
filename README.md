# Eye Disease Screening — AI for retina (fundus) photos

Final-year capstone project · Dayananda Sagar University · Computer Science & Engineering
Team: Bhanu Shankar Prajwal P, Tejas C E · Guide: Dr. Revathi V

A web app that looks at a retina (fundus) photo and checks it for **8 eye conditions**:
Normal, Diabetic retinopathy, Glaucoma, Cataract, AMD, Hypertensive retinopathy, Myopia and Others.
It also shows **where in the eye the model looked** (Grad-CAM heatmap) and can **sort many photos by urgency**.

> Research prototype only. It is a screening aid, not a diagnosis, and it is not validated for real patient use.

---

## 1. Run it on your laptop (Windows)

**You need:** Python **3.10, 3.11 or 3.12** (from [python.org](https://www.python.org/downloads/) — tick
*"Add Python to PATH"* during install). No GPU needed; it runs on a normal laptop CPU.

### Easiest way — double-click
1. Download this repository (green **Code** button → **Download ZIP**) and unzip it, or `git clone` it.
2. Double-click **`run_app.bat`**.
   - The first time, it installs everything it needs (5–10 minutes, needs internet).
   - After that it starts in a few seconds.
3. The app opens in your browser at **http://localhost:8501**. To stop it, close the black window.

### Or with commands (any system)
```bash
cd eye-disease-screening
python -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
streamlit run Code/app.py
```

---

## 2. What you can do in the app

Use the menu on the left.

| Page | What it does |
|---|---|
| **Home** | Short introduction, links to the tools, the main test scores. |
| **Single photo check** | Upload one or both eyes (or pick a sample photo) → the 8 conditions ranked, a clear result box (Refer / Doctor review / Likely normal) and the Grad-CAM heatmap. |
| **Myopia check** | Age + sex + photo → chance of Myopia, with a heatmap. |
| **Batch screening** | Many photos at once (several files, a ZIP, or a folder) → sorted into Refer / Doctor review / Likely normal / Retake photo. Download a CSV table, a printable report or a ZIP of the sorted photos. |
| **About & results** | The data, the models and the real test results, and the limitations. |

**Quick test:** the folder `test_images/` has 16 photos the model never saw during training. The true answer is in
each file name (e.g. `07_Cataract_999_right.jpg`). Use them in *Single photo check* (“Use a sample test photo”) or
point *Batch screening* → *Folder on this computer* at the `test_images` folder.

---

## 3. Folder map

```
eye-disease-screening/
├── run_app.bat              ← double-click to install + start the app (Windows)
├── requirements.txt         ← Python packages
├── README.md                ← this file
├── .streamlit/config.toml   ← app colours (dark theme)
│
├── Code/
│   ├── app.py               ← START HERE: the web app (menu + pages)
│   ├── views/               ← one file per page: home, screen (single photo), myopia, batch, about
│   ├── ui.py                ← shared look: colours, cards, result boxes
│   ├── batch_ui.py          ← batch screening page layout
│   ├── assets/              ← logo
│   │
│   ├── predict_multiclass.py   ← 8-condition model: loads it and predicts
│   ├── predict_myopia.py       ← Myopia model (age + sex + photo)
│   ├── gradcam.py              ← heatmaps (where the model looked)
│   ├── triage.py               ← batch sorting logic (also works from the terminal)
│   ├── image_quality.py        ← photo-quality check used by batch screening
│   │
│   ├── models/              ← trained model files (.pt) used by the app
│   ├── training/            ← scripts used to train / calibrate the models (need the full datasets)
│   └── notebooks/           ← step-by-step notebooks of the whole project + result charts
│
├── Data/                    ← model settings, test results and logs (small files the app reads)
├── test_images/             ← 16 held-out sample photos for quick testing
└── docs/                    ← full project documentation (Word), the project story (PDF) and review notes
```

---

## 4. Models and results (all on photos the model never saw)

| Part | Model | Result |
|---|---|---|
| 8 conditions | EfficientNet-B0 (trained on ODIR-5K + RFMiD) + DenseNet-121, averaged | 72.7% accuracy · 0.666 macro-F1 · mean AUC 0.928 on 960 ODIR test photos · mean AUC 0.771 on 640 RFMiD test photos |
| Myopia check | MobileNetV2 image branch + age/sex, fused | 97.4% accuracy · 84% of Myopia cases found · 504 test patients |
| Batch screening | Rules on top of the 8-condition model, limits set on validation photos | 94.6% of diseased eyes sent to a doctor · 4.3% missed · 91.0% of “Refer” photos really diseased (1,600 test photos) |

**Comparison with other networks:** 7 well-known networks (MobileNetV2, EfficientNet-B0, ResNet-50, DenseNet-121,
VGG-16, ConvNeXt-Tiny, ViT-B/16) were trained on the same split with the same settings. Our final model is in the
top group (best macro-F1 and AUC); results in `Data/benchmark_results.csv`, learning curve in
`Data/learning_curve_results.csv`, model combinations in `Data/ensemble_search.csv`.

Full details, every experiment and the honest limitations are in `docs/Capstone_Project_Documentation.docx`
and on the **About & results** page of the app.

**Data used:** ODIR-5K (6,392 photos, 3,358 patients, split by patient so no person is in both training and test)
and RFMiD (Pachade et al., *Data* 2021, CC BY 4.0). The raw image datasets are **not** included here (too large);
the app does not need them. They are only needed to re-run the training scripts.

---

## 5. Common problems

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python from python.org and tick *Add Python to PATH*, then open a new window. |
| Install fails on `torch` | Use Python 3.10–3.12 (not 3.13+). |
| Page shows “model could not be loaded” | Make sure the `Code/models` folder has the `.pt` files (download the full repo, not single files). |
| Port 8501 already in use | Close the other app window, or run `streamlit run Code/app.py --server.port 8502`. |
| First prediction is slow | Normal — the models load once (10–30 s on CPU), then it is fast. |
