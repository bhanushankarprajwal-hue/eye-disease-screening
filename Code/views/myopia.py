"""Myopia check: age + sex + fundus photo (Stage 8 fusion model)."""
import tempfile

import streamlit as st

import gradcam as gc
import predict_multiclass as pmc
import predict_myopia as pm
import ui


def render():
    ui.page_header("Myopia check",
                   "Combines the patient's age and sex with a fundus photo to estimate the chance of Myopia "
                   "(short-sightedness).", "glasses")
    config, bundle = ui.load_or_stop(ui.myopia_model)

    ui.step(1, "Patient details")
    c1, c2 = st.columns(2, gap="medium")
    age = c1.number_input("Age (years)", min_value=1, max_value=110, value=40, step=1, key="my_age")
    sex = c2.segmented_control("Sex", ["Male", "Female"], default="Male", key="my_sex") or "Male"

    ui.step(2, "Choose a photo")
    samples = ui.sample_images()
    options = ["Upload my own photo"] + (["Use a sample test photo"] if samples else [])
    src = st.segmented_control("Photo source", options, default=options[0], key="my_src",
                               label_visibility="collapsed")
    chosen = []
    if src == "Use a sample test photo":
        label = st.selectbox("Sample photo (from the held-out test set)", list(samples), key="my_sample",
                             index=next((i for i, k in enumerate(samples) if k.startswith("Myopia")), 0))
        chosen = [samples[label]]
        st.image(samples[label], width=260)
    else:
        u1, u2 = st.columns(2, gap="medium")
        f1 = u1.file_uploader("Eye photo 1", type=["jpg", "jpeg", "png"], key="my_f1")
        f2 = u2.file_uploader("Eye photo 2 (optional — the other eye)", type=["jpg", "jpeg", "png"], key="my_f2")
        if f1 is not None:
            u1.image(f1, width=260)
        if f2 is not None:
            u2.image(f2, width=260)
        chosen = [f for f in (f1, f2) if f is not None]

    ui.step(3, "Check")
    run = st.button("Check for Myopia", type="primary", icon=":material/play_arrow:", disabled=not chosen,
                    use_container_width=True, key="my_run")
    if not chosen:
        st.caption("Choose a photo above to enable this button.")

    key = (age, sex) + tuple(getattr(p, "file_id", p) for p in chosen)
    if run:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [p if isinstance(p, str) else ui.save_uploads([p], tmp)[0] for p in chosen]
            with st.spinner("Checking..."):
                prob, label, threshold, _ = pm.predict_with_loaded(bundle, config, age, sex, paths[0],
                                                                   paths[1] if len(paths) > 1 else None)
                heat, _ = gc.myopia_cam(config, bundle, paths[0], age, sex)
                first = pmc.load_rgb(paths[0])
        st.session_state["my_result"] = {"key": key, "prob": prob, "label": label, "threshold": threshold,
                                         "heat": heat, "first": first, "age": age, "sex": sex, "n": len(paths)}

    res = st.session_state.get("my_result")
    if not res or res["key"] != key:
        ui.footer()
        return
    st.divider()
    st.subheader("Result")
    is_m = res["label"] == "Myopia"
    ui.result_banner("refer" if is_m else "normal",
                     "Myopia likely" if is_m else "Myopia not likely",
                     f"Chance of Myopia: {res['prob']:.0%} (decision limit {res['threshold']:.0%}) · "
                     f"patient: {res['age']} years, {res['sex'].lower()}<br>"
                     + ("Please confirm with an eye test by an optometrist or eye doctor." if is_m else
                        "The model did not find signs of Myopia in this photo."),
                     pill_text="Myopia" if is_m else "No Myopia")
    if res["n"] == 2:
        st.caption("Both photos were combined (their results were averaged).")
    t1, t2 = st.tabs(["Where the model looked", "How reliable is this?"])
    with t1:
        c1, c2, _ = st.columns([1, 1, 0.5], gap="medium")
        c1.image(res["first"], caption="Photo (colours corrected if needed)", use_container_width=True)
        c2.image(res["heat"], caption="Where the model looked for Myopia (Grad-CAM)", use_container_width=True)
        st.caption("Red/yellow areas pushed the model most towards Myopia; blue areas were mostly ignored.")
    with t2:
        m = config["test_metrics"]
        k1, k2, k3 = st.columns(3, gap="medium")
        with k1:
            ui.kpi(f"{m['accuracy']*100:.1f}%", "Accuracy on 504 unseen patients", "target", "#3B82F6")
        with k2:
            ui.kpi(f"{m['recall_myopia']*100:.0f}%", "Myopia cases found (recall)", "eye", "#14B8A6")
        with k3:
            ui.kpi(f"{m['precision_myopia']*100:.0f}%", "Correct when it says Myopia (precision)", "shield", "#F59E0B")
        st.caption("Age and sex add only a little information; the photo carries most of the signal. "
                   "Myopia is rare in this data (about 4% of patients), which is why precision is lower than recall.")
    ui.footer()


render()
