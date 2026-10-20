"""Single photo check: 8 conditions from one or two fundus photos."""
import os
import tempfile

import pandas as pd
import streamlit as st

import gradcam as gc
import predict_multiclass as pmc
import ui

NEXT_STEP = {
    "refer": "The model suggests an eye condition. Please confirm with an eye doctor.",
    "review": "The model is not sure about this photo. A doctor should look at it.",
    "normal": "The model did not find signs of the 8 conditions it knows. Routine check-ups are still advised.",
}


def render():
    ui.page_header("Single photo check",
                   "Upload a fundus photo of one or both eyes. The model ranks 8 conditions and shows where it looked.", "scan")
    config, model, device = ui.load_or_stop(ui.multiclass_model)

    ui.step(1, "Choose a photo")
    samples = ui.sample_images()
    options = ["Upload my own photo"] + (["Use a sample test photo"] if samples else [])
    src = st.segmented_control("Photo source", options, default=options[0], key="sc_src",
                               label_visibility="collapsed")
    paths_or_files = []
    if src == "Use a sample test photo":
        label = st.selectbox("Sample photo (from the held-out test set — the model never trained on these)",
                             list(samples), key="sc_sample")
        paths_or_files = [samples[label]]
        st.image(samples[label], width=260)
    else:
        c1, c2 = st.columns(2, gap="medium")
        f1 = c1.file_uploader("Eye photo 1", type=["jpg", "jpeg", "png"], key="sc_f1")
        f2 = c2.file_uploader("Eye photo 2 (optional — the other eye)", type=["jpg", "jpeg", "png"], key="sc_f2")
        if f1 is not None:
            c1.image(f1, width=260)
        if f2 is not None:
            c2.image(f2, width=260)
        paths_or_files = [f for f in (f1, f2) if f is not None]

    ui.step(2, "Check")
    run = st.button("Check photo" if len(paths_or_files) < 2 else "Check both photos", type="primary",
                    icon=":material/play_arrow:", disabled=not paths_or_files, use_container_width=True, key="sc_run")
    if not paths_or_files:
        st.caption("Choose a photo above to enable this button.")

    key = tuple(getattr(p, "file_id", p) for p in paths_or_files)
    if run:
        with tempfile.TemporaryDirectory() as tmp:
            paths = [p if isinstance(p, str) else ui.save_uploads([p], tmp)[0] for p in paths_or_files]
            with st.spinner("Checking the photo..."):
                ranked = pmc.predict_multiclass(config, model, device, paths)
                first = pmc.load_rgb(paths[0])
                heat = [(name, gc.multiclass_cam(config, model, device, paths[0], name)[0]) for name, _ in ranked[:2]]
        st.session_state["sc_result"] = {"key": key, "ranked": ranked, "first": first, "heat": heat,
                                         "n": len(paths)}
    res = st.session_state.get("sc_result")
    if not res or res["key"] != key:
        ui.footer()
        return

    ranked = res["ranked"]
    status, limits = ui.triage_status(ranked)
    top, top_p = ranked[0]
    second, second_p = ranked[1]
    st.divider()
    st.subheader("Result")
    title = ("Likely normal" if status == "normal" else
             f"Most likely: {top}" if status == "refer" else f"Unsure — leaning {top}")
    ui.result_banner(status, title,
                     f"Confidence {top_p:.0%} · next most likely: {second} ({second_p:.0%})<br>{NEXT_STEP[status]}")
    if res["n"] == 2:
        st.caption("Both photos were combined (their results were averaged).")

    t1, t2, t3 = st.tabs(["All 8 conditions", "Where the model looked", "How reliable is this?"])
    with t1:
        ui.prob_bars(ranked)
        st.caption(f"Rule used for the label above (same as batch screening): Refer if a disease has at least "
                   f"{limits['refer_min_confidence']:.0%} confidence; Likely normal if Normal has at least "
                   f"{limits['normal_min_confidence']:.0%}; otherwise Doctor review.")
    with t2:
        cols = st.columns(len(res["heat"]) + 1, gap="medium")
        cols[0].image(res["first"], caption="Photo (colours corrected if needed)", use_container_width=True)
        for col, (name, img) in zip(cols[1:], res["heat"]):
            col.image(img, caption=f"Where it looked for {name}", use_container_width=True)
        st.caption("Grad-CAM heatmap: red/yellow areas pushed the model most towards that answer; blue areas were mostly "
                   "ignored. It explains the model's decision — it is not a medical marking of the disease. "
                   "Shown for the first photo.")
    with t3:
        pc = config.get("per_class_csv")
        path = os.path.join(pmc.PROJECT_ROOT, pc) if pc else None
        if path and os.path.isfile(path):
            df = pd.read_csv(path)
            row = df.set_index("class").loc[top]
            st.markdown(f"On the held-out test set, when the true answer was **{top}**, the model found it "
                        f"**{row.cnn_recall:.0%}** of the time, and when it said **{top}** it was right "
                        f"**{row.cnn_precision:.0%}** of the time.")
            st.dataframe(pd.DataFrame({"Condition": df["class"], "Test photos": df.test_images,
                                       "Found (recall)": (df.cnn_recall * 100).round(0),
                                       "Correct when said (precision)": (df.cnn_precision * 100).round(0),
                                       "F1": (df.cnn_f1 * 100).round(0)}),
                         hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")
                                        for c in ["Found (recall)", "Correct when said (precision)", "F1"]})
        else:
            st.info("Per-condition test results are not available for this model.")
    ui.footer()


render()
