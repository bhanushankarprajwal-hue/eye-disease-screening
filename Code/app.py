"""
Eye Disease Screening -- web app for the capstone project.

Pages (sidebar):
  Home                 -- what the app does, model summary
  Single photo check   -- 8 conditions from one or two photos, with Grad-CAM   (views/screen.py)
  Myopia check         -- age + sex + photo, Stage 8 fusion model               (views/myopia.py)
  Batch screening      -- many photos sorted by urgency                         (views/batch.py, batch_ui.py, triage.py)
  About                -- data, models, real test results, limitations          (views/about.py)

Run (from the project root, with venv311 active):
    streamlit run Code/app.py
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402
import predict_multiclass as pmc  # noqa: E402

st.set_page_config(page_title="Eye Disease Screening", page_icon=":material/visibility:", layout="wide",
                   initial_sidebar_state="expanded")
ui.apply_style()

nav = st.navigation({
    "": [st.Page("views/home.py", title="Home", icon=":material/home:", default=True)],
    "Tools": [
        st.Page("views/screen.py", title="Single photo check", icon=":material/image_search:"),
        st.Page("views/myopia.py", title="Myopia check", icon=":material/visibility:"),
        st.Page("views/batch.py", title="Batch screening", icon=":material/photo_library:"),
    ],
    "Project": [st.Page("views/about.py", title="About & results", icon=":material/insights:")],
})

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
st.logo(os.path.join(ASSETS, "logo.svg"), size="large", icon_image=os.path.join(ASSETS, "icon.svg"))

nav.run()

with st.sidebar:
    import torch
    try:
        model_name = pmc.describe(pmc.pick_config())
    except Exception:  # noqa: BLE001
        model_name = "not found"
    run_on = f"GPU · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    st.markdown(f'<div class="side-info"><b>Model</b><br>{model_name}<br><br><b>Running on</b><br>{run_on}<br><br>'
                'Capstone project · Dayananda Sagar University<br>Research prototype — not for clinical use.</div>',
                unsafe_allow_html=True)
