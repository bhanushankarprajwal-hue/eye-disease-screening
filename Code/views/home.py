"""Home page: what the app does and where to go."""
import json
import os

import streamlit as st

import predict_multiclass as pmc
import ui


def render():
    ui.hero("AI fundus screening · capstone project", "Eye Disease Screening",
            "Checks retina (fundus) photos for 8 eye conditions with deep learning, shows where the model looked, "
            "and sorts many photos by how urgently a doctor should see them.")

    st.subheader("What would you like to do?")
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        ui.card("Check one photo",
                "Upload one or both eyes. The model ranks 8 conditions and shows where in the eye it looked.",
                "scan", "#3B82F6")
        st.page_link("views/screen.py", label="Open single photo check", icon=":material/arrow_forward:")
    with c2:
        ui.card("Myopia check",
                "Combines the patient's age and sex with the photo to estimate the chance of Myopia.",
                "glasses", "#14B8A6")
        st.page_link("views/myopia.py", label="Open Myopia check", icon=":material/arrow_forward:")
    with c3:
        ui.card("Batch screening",
                "Screens many photos at once and sorts them into Refer, Doctor review, Likely normal and Retake.",
                "layers", "#F59E0B")
        st.page_link("views/batch.py", label="Open batch screening", icon=":material/arrow_forward:")

    st.subheader("Model at a glance")
    cfg = pmc.pick_config()
    m = cfg.get("test_metrics", {})
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        ui.kpi(f"{m.get('accuracy', 0)*100:.1f}%", "Accuracy, 960 unseen test photos", "target", "#3B82F6")
    with k2:
        ui.kpi(f"{m.get('f1_macro', 0)*100:.1f}%", "Macro-F1 (average over 8 classes)", "chart", "#14B8A6")
    with k3:
        ui.kpi(f"{m.get('mean_auc', 0):.3f}", "Mean AUC (1.0 = perfect ranking)", "shield", "#A78BFA")
    with k4:
        rf = m.get("rfmid_mean_auc")
        ui.kpi(f"{rf:.3f}" if rf else "—", "Mean AUC on a second dataset (RFMiD)", "globe", "#F59E0B")
    st.write("")
    st.caption(f"Model in use: {pmc.describe(cfg)}. Full results on the About & results page.")

    st.subheader("Conditions covered")
    ui.chips([("Normal", "#22C55E"), ("Diabetic retinopathy", "#EF4444"), ("Glaucoma", "#A78BFA"),
              ("Cataract", "#94A3B8"), ("Age-related macular degeneration (AMD)", "#F59E0B"),
              ("Hypertensive retinopathy", "#F472B6"), ("Myopia", "#14B8A6"), ("Others", "#3B82F6")])
    ui.footer()


render()
