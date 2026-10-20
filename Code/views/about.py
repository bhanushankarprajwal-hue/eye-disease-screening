"""About: data, models, real test results and limitations (all numbers read from the project's result files)."""
import json
import os

import pandas as pd
import streamlit as st

import predict_multiclass as pmc
import ui

ROOT = pmc.PROJECT_ROOT
MISSED = "diseased eyes wrongly in 'Likely normal' %"
CLEARED = "healthy eyes cleared as 'Likely normal' %"


def _csv(name):
    p = os.path.join(ROOT, "Data", name)
    return pd.read_csv(p) if os.path.isfile(p) else None


def _pct(cols):
    return {c: st.column_config.NumberColumn(format="%.1f%%") for c in cols}


def render():
    ui.page_header("About this project",
                   "How the system was built and how well it really works. Every number below comes from the "
                   "project's result files, measured on photos the models never saw during training.", "chart")

    t1, t2, t3, t4 = st.tabs(["Data", "8-condition model", "Batch screening", "Limitations"])

    with t1:
        st.markdown("""
**ODIR-5K** (Ocular Disease Intelligent Recognition) — 6,392 labelled fundus photos from 3,358 patients,
8 classes. Split **by patient** into training / validation / test (no patient appears in two parts), so the
test results are on new people.

**RFMiD** (Retinal Fundus Multi-Disease Image Dataset; Pachade et al., 2021, CC BY 4.0) — 3,200 photos from a
different eye clinic and different cameras. Used first as an outside test, then partly for training so the
model works on more than one kind of camera.

**Clinical data** — only age and sex are available. Diagnosis keywords and label columns were removed to
avoid answer leakage.
""")
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            ui.kpi("6,392", "ODIR photos (3,358 patients)", "eye", "#3B82F6")
        with c2:
            ui.kpi("3,200", "RFMiD photos (second clinic)", "globe", "#14B8A6")
        with c3:
            ui.kpi("960 + 640", "Unseen test photos (ODIR + RFMiD)", "target", "#F59E0B")

    with t2:
        cfg = pmc.pick_config()
        st.markdown(f"**Model in use:** {pmc.describe(cfg)} — two EfficientNet-B0 networks (ImageNet pre-trained, "
                    "fine-tuned on fundus photos) whose answers are averaged. Each photo is also checked mirrored. "
                    "Chosen on the validation set only.")
        comp = _csv("stage13_comparison.csv")
        if comp is not None:
            show = pd.DataFrame({
                "Model": comp["model"],
                "ODIR accuracy": comp["ODIR test accuracy"] * 100,
                "ODIR macro-F1": comp["ODIR test macro-F1"] * 100,
                "ODIR mean AUC": comp["ODIR test mean AUC"],
                "RFMiD mean AUC": comp["RFMiD mean AUC"],
            })
            st.markdown("**Improvement over the project (test photos)**")
            st.dataframe(show, hide_index=True, use_container_width=True,
                         column_config={**_pct(["ODIR accuracy", "ODIR macro-F1"]),
                                        "ODIR mean AUC": st.column_config.NumberColumn(format="%.3f"),
                                        "RFMiD mean AUC": st.column_config.NumberColumn(format="%.3f")})
            st.caption("The first 8-class model (MobileNetV2, 224px) reached 49.8% accuracy; a model that always says "
                       "'Normal' would score 7.8% macro-F1.")
        pc = _csv(os.path.basename(cfg.get("per_class_csv", ""))) if cfg.get("per_class_csv") else None
        if pc is not None:
            st.markdown("**Per condition (960 ODIR test photos)**")
            st.dataframe(pd.DataFrame({"Condition": pc["class"], "Test photos": pc.test_images,
                                       "Found (recall)": pc.cnn_recall * 100,
                                       "Correct when said (precision)": pc.cnn_precision * 100,
                                       "F1": pc.cnn_f1 * 100}),
                         hide_index=True, use_container_width=True,
                         column_config=_pct(["Found (recall)", "Correct when said (precision)", "F1"]))
        st.markdown("**Explainability:** every answer can be shown as a Grad-CAM heatmap of the area the model used.")

    with t3:
        ev = _csv("triage_evaluation.csv")
        lim_file = os.path.join(ROOT, "Data", "triage_thresholds.json")
        if os.path.isfile(lim_file):
            with open(lim_file) as f:
                st.markdown(f"**Decision limits:** {json.load(f).get('summary', '')}")
        if ev is not None:
            s = ev[(ev.images == "TEST both") & (ev.group == "SUMMARY")].iloc[0]
            r = ev[(ev.images == "TEST both") & (ev.group == "Refer")].iloc[0]
            c1, c2, c3, c4 = st.columns(4, gap="medium")
            with c1:
                ui.kpi(f"{s['diseased eyes sent to Refer or Review %']:.1f}%", "Diseased eyes sent to a doctor", "shield", "#22C55E")
            with c2:
                ui.kpi(f"{r['really diseased %']:.1f}%", "Refer photos that are really diseased", "target", "#EF4444")
            with c3:
                ui.kpi(f"{s[MISSED]:.1f}%", "Diseased eyes wrongly in Likely normal", "eye", "#F59E0B")
            with c4:
                ui.kpi(f"{s[CLEARED]:.1f}%", "Healthy eyes cleared as Likely normal", "chart", "#3B82F6")
            g = ev[(ev.images == "TEST both") & (ev.group != "SUMMARY")]
            st.markdown("**Groups on 1,600 unseen test photos (ODIR + RFMiD)**")
            st.dataframe(pd.DataFrame({"Group": g.group, "Photos": g.photos, "Share of photos": g["share of photos %"],
                                       "Really diseased": g["really diseased %"]}),
                         hide_index=True, use_container_width=True,
                         column_config=_pct(["Share of photos", "Really diseased"]))
        img = os.path.join(ROOT, "Code", "notebooks", "triage_evaluation.png")
        if os.path.isfile(img):
            st.image(img, use_container_width=True)

    with t4:
        st.markdown("""
- **Accuracy is about 72%** for the exact condition. Hypertension and Glaucoma are the weakest classes
  (few training photos, subtle signs).
- One label per photo: a real eye can have more than one condition at once.
- Age and sex carry little information; real clinical measurements (refraction, eye pressure) are not available.
- After training on RFMiD, a truly unseen-hospital test needs a third dataset.
- The photo-quality check is rule-based and catches only clear problems (very dark or over-exposed photos).
- "Likely normal" means lower priority, **not** "no doctor needed" — about 5% of diseased eyes land there.
- Research prototype only: not a medical device and not validated for real patient care.
""")
    ui.footer()


render()
