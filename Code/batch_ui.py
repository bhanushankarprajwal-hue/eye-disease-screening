"""
"Batch screening" page of the web app (Code/app.py): screen many fundus photos at once and sort them
into Refer / Doctor review / Likely normal / Retake photo. All the logic lives in Code/triage.py;
this file is only the user interface.
"""
import base64
import datetime as dt
import html
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import gradcam as gc
import predict_multiclass as pmc
import triage as tr
import ui

COLORS = {g: ui.STATUS[g]["color"] for g in ("refer", "review", "normal", "retake")}
MAX_ZIP_IMAGES = 5000

CSS = """
<style>
.block-container { max-width: 1180px !important; }   /* wider page for the results table */
.tri-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.7rem; margin:0.4rem 0 1.2rem 0; }
@media (max-width: 760px) { .tri-grid { grid-template-columns:repeat(2,1fr); } }
.tri-card { border:1px solid rgba(255,255,255,0.08); border-left-width:5px; border-radius:14px; padding:0.85rem 1rem;
            box-shadow: 0 8px 22px rgba(0,0,0,0.25); }
.tri-card .n { font-size:1.8rem; font-weight:700; line-height:1.15; }
.tri-card .l { font-size:0.95rem; font-weight:650; }
.tri-card .s { font-size:0.8rem; opacity:0.72; margin-top:0.2rem; line-height:1.35; }
.tri-meta { font-size:0.88rem; opacity:0.72; margin-bottom:0.6rem; }
.legend { display:flex; gap:1rem; font-size:0.82rem; opacity:0.8; margin:0.2rem 0 0.4rem 0; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:0.35rem; vertical-align:middle; }
</style>
"""


# ------------------------------------------------------------------ helpers
def _thumb_uri(src, size=96):
    try:
        img = tr._open(src)
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _read_zip(uploaded):
    items = []
    with zipfile.ZipFile(uploaded) as z:
        for info in z.infolist():
            name = info.filename
            base = os.path.basename(name)
            if info.is_dir() or base.startswith(".") or "__MACOSX" in name or not base.lower().endswith(tr.IMAGE_EXT):
                continue
            items.append((name, z.read(info)))
            if len(items) >= MAX_ZIP_IMAGES:
                break
    return sorted(items, key=lambda t: t[0])


def _src_bytes(src):
    if isinstance(src, (bytes, bytearray)):
        return bytes(src)
    with open(src, "rb") as f:
        return f.read()


def _cards(df):
    n = max(len(df), 1)
    cards = []
    for g, (_o, short, _long, advice) in tr.GROUPS.items():
        k = int((df.group == g).sum())
        cards.append(f'<div class="tri-card" style="border-left-color:{COLORS[g]};background:linear-gradient(135deg,{ui.tint(COLORS[g], 0.16)},#13223A 75%)">'
                     f'<div class="n" style="color:{COLORS[g]}">{k}</div>'
                     f'<div class="l">{short} <span style="opacity:0.6;font-weight:500">· {100*k/n:.0f}%</span></div>'
                     f'<div class="s">{advice}</div></div>')
    return '<div class="tri-grid">' + "".join(cards) + "</div>"


def _findings_html(df):
    """Horizontal bars: main finding of the photos that need a doctor (Refer in red, Doctor review in amber)."""
    d = df[df.group.isin(["refer", "review"]) & (df.top_finding != "")]
    if d.empty:
        return ""
    c = d.groupby(["top_finding", "group"]).size().unstack(fill_value=0)
    for g in ("refer", "review"):
        if g not in c:
            c[g] = 0
    c["total"] = c.refer + c.review
    c = c.sort_values(["total", "refer"], ascending=False)
    mx = max(int(c.total.max()), 1)
    rows = []
    for name, r in c.iterrows():
        label = "Normal (unsure)" if name == "Normal" else str(name)
        rows.append(
            f'<div class="bar-row"><div class="bar-name" style="width:130px">{html.escape(label)}</div>'
            f'<div class="bar-track">'
            f'<div style="width:{100*r.refer/mx:.1f}%;background:{COLORS["refer"]}"></div>'
            f'<div style="width:{100*r.review/mx:.1f}%;background:{COLORS["review"]}"></div></div>'
            f'<div class="bar-val" style="width:150px">{int(r.refer)} refer · {int(r.review)} review</div></div>')
    legend = (f'<div class="legend"><span><i style="background:{COLORS["refer"]}"></i>Refer</span>'
              f'<span><i style="background:{COLORS["review"]}"></i>Doctor review</span></div>')
    return legend + "".join(rows)


def _report_html(res):
    df = res["df"]
    n = len(df)
    rows = []
    for r in df.itertuples():
        thumb = res["thumbs"].get(r.file)
        img = f'<img src="{thumb}" width="64">' if thumb else ""
        conf = f"{r.confidence:.0%}" if r.top_finding else ""
        second = f"{r.second_finding} {r.second_confidence:.0%}" if r.top_finding else ""
        rows.append(f"<tr><td>{r.priority}</td><td>{img}</td><td>{html.escape(r.file)}</td>"
                    f'<td><span class="pill" style="background:{COLORS[r.group]}">{tr.GROUPS[r.group][1]}</span></td>'
                    f"<td>{html.escape(r.top_finding)} {conf}</td><td>{html.escape(second)}</td>"
                    f"<td>{html.escape(r.quality_notes)}</td><td>{html.escape(r.reason)}</td></tr>")
    counts = "".join(f'<div class="card" style="border-left-color:{COLORS[g]}"><b style="color:{COLORS[g]}">{int((df.group == g).sum())}</b><br>{tr.GROUPS[g][1]}</div>'
                     for g in tr.GROUPS)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Batch screening report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#1d2330}} h1{{margin:0 0 4px 0}}
.meta{{color:#5b6475;font-size:13px}} .cards{{display:flex;gap:10px;margin:16px 0}}
.card{{flex:1;border:1px solid #dde1e7;border-left-width:5px;border-radius:8px;padding:10px 12px;font-size:14px}} .card b{{font-size:24px}}
table{{border-collapse:collapse;width:100%;font-size:12.5px}} th,td{{border-bottom:1px solid #e3e6eb;padding:5px 6px;text-align:left;vertical-align:middle}}
th{{background:#f3f5f8}} .pill{{color:#fff;border-radius:999px;padding:2px 8px;font-weight:600;white-space:nowrap}}
.note{{margin-top:18px;font-size:12px;color:#5b6475}} @media print{{body{{margin:10mm}}}}</style></head><body>
<h1>Batch screening report</h1>
<div class="meta">{html.escape(res['when'])} · {n} photos · {html.escape(res['model'])}<br>Decision limits: {html.escape(res['limits'])}</div>
<div class="cards">{counts}</div>
<table><tr><th>#</th><th>Photo</th><th>File</th><th>Group</th><th>Main finding</th><th>Next most likely</th><th>Photo quality</th><th>Why this group</th></tr>
{''.join(rows)}</table>
<div class="note">Research prototype built for a final-year capstone project (models trained on the public ODIR-5K and RFMiD
datasets). This is a screening aid that sorts photos for a doctor; it is not a diagnosis and not validated for real patient use.</div>
</body></html>"""


def _sorted_zip(res):
    buf = io.BytesIO()
    df = res["df"]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for r in df.itertuples():
            o, short = tr.GROUPS[r.group][0], tr.GROUPS[r.group][1].replace(" ", "_")
            try:
                data = _src_bytes(res["sources"][r.file])
            except Exception:  # noqa: BLE001
                continue
            z.writestr(f"{o}_{short}/{r.priority:04d}_{os.path.basename(r.file)}", data)
        z.writestr("triage_results.csv", df.drop(columns=[c for c in df.columns if c.startswith("p_") and c != "p_disease"])
                   .to_csv(index=False).encode("utf-8-sig"))
        z.writestr("report.html", _report_html(res))
    return buf.getvalue()


# ------------------------------------------------------------------ page
def render(config, model, device):
    st.markdown(CSS, unsafe_allow_html=True)
    limits, limits_info = tr.load_thresholds()
    with st.expander("How the sorting works", icon=":material/info:"):
        st.markdown(
            f"""
1. **Photo quality check** (no AI). Too dark, over-exposed, almost no fundus visible, or probably not a fundus
   photo → **Retake photo**. Blur and uneven lighting are only shown as notes, because a cataract itself makes a
   photo look hazy and such eyes must not be hidden.
2. **AI check** of every usable photo (8 diseases), then:
   - **Refer** — the top finding is a disease with confidence ≥ **{limits['refer_min_confidence']:.0%}**
   - **Likely normal** — the top finding is Normal with confidence ≥ **{limits['normal_min_confidence']:.0%}**
   - **Doctor review** — everything in between (the model is not sure)
3. Inside each group the photos are ordered by urgency (most likely disease first).

**Where the limits come from:** {limits_info}
""")
    ui.step(1, "Choose the photos")
    src = st.segmented_control("Source", ["Upload photos", "Upload a ZIP file", "Folder on this computer"],
                               default="Upload photos", key="batch_src", label_visibility="collapsed")
    items = []
    if src == "Upload photos":
        files = st.file_uploader("Fundus photos — select many at once (JPG or PNG)", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="batch_files")
        items = [(f.name, f.getvalue()) for f in (files or [])]
    elif src == "Upload a ZIP file":
        zf = st.file_uploader("ZIP file with fundus photos (sub-folders are fine)", type=["zip"], key="batch_zip")
        if zf is not None:
            try:
                items = _read_zip(zf)
            except zipfile.BadZipFile:
                st.error("That file is not a valid ZIP file.")
    elif src == "Folder on this computer":
        default = os.path.join(pmc.PROJECT_ROOT, "test_images")
        folder = st.text_input("Folder path (copy it from the File Explorer address bar)", value=default,
                               key="batch_folder").strip().strip('"')
        if folder:
            if os.path.isdir(folder):
                items = [(os.path.relpath(p, folder), p) for p in tr.list_images(folder)]
                if not items:
                    st.warning("No JPG or PNG photos found in that folder.")
            else:
                st.error("Folder not found. Check the path and try again.")

    by_patient = st.checkbox("Also group results by patient (for file names like 123_left.jpg and 123_right.jpg)",
                             value=True, key="batch_by_patient")
    ui.step(2, "Screen")
    run = st.button(f"Screen {len(items)} photo{'s' if len(items) != 1 else ''}" if items else "Screen photos",
                    type="primary", icon=":material/play_arrow:", use_container_width=True, disabled=not items,
                    key="batch_run")
    if not items:
        st.caption("Add photos above to enable this button.")

    if run and items:
        names = [n for n, _ in items]
        if len(set(names)) != len(names):   # make file names unique (same name in different sub-folders)
            seen = {}
            fixed = []
            for n, s in items:
                k = seen.get(n, 0)
                seen[n] = k + 1
                fixed.append((n if k == 0 else f"{os.path.splitext(n)[0]} ({k}){os.path.splitext(n)[1]}", s))
            items = fixed
        bar = st.progress(0.0, text="Checking photo quality...")
        t0 = time.time()

        def progress(done, n, eta):
            bar.progress(done / n, text=f"Screening {done}/{n} photos · about {eta:.0f}s left")

        df = tr.screen(config, device, items, progress=progress, thresholds=limits)
        bar.progress(1.0, text="Making thumbnails...")
        sources = dict(items)
        thumbs = {n: _thumb_uri(s) for n, s in items}
        took = time.time() - t0
        bar.empty()
        st.session_state["batch_result"] = {
            "df": df, "sources": sources, "thumbs": thumbs, "took": took, "by_patient": by_patient,
            "model": pmc.describe(config), "limits": limits_info,
            "when": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
        }
        st.session_state.pop("batch_cam", None)

    res = st.session_state.get("batch_result")
    if not res:
        return
    df = res["df"]
    st.divider()
    st.subheader("Results")
    st.markdown(f'<div class="tri-meta">{len(df)} photos screened in {res["took"]:.1f} s '
                f'({res["took"]/max(len(df),1):.2f} s per photo) · {html.escape(res["when"])}</div>', unsafe_allow_html=True)
    st.markdown(_cards(df), unsafe_allow_html=True)
    fh = _findings_html(df)
    if fh:
        st.markdown("**Main finding of the photos that need a doctor**")
        st.markdown(fh, unsafe_allow_html=True)

    two_eyes = res["by_patient"] and bool((df.eye != "").any()) and df.patient.duplicated().any()
    res["show_patients"] = two_eyes
    tab_names = ["Photos (by urgency)"] + (["Patients (both eyes)"] if two_eyes else []) + ["Download"]
    tabs = st.tabs(tab_names)

    # ---------------------------------------------------------------- photos table
    with tabs[0]:
        f1, f2 = st.columns([3, 2])
        show = f1.multiselect("Show groups", options=list(tr.GROUPS), default=list(tr.GROUPS),
                              format_func=lambda g: tr.GROUPS[g][1], key="batch_show")
        q = f2.text_input("Search by file name", key="batch_q", placeholder="e.g. 123_left").strip().lower()
        view = df[df.group.isin(show)]
        if q:
            view = view[view.file.str.lower().str.contains(q, regex=False)]
        table = pd.DataFrame({
            "#": view.priority,
            "Photo": view.file.map(res["thumbs"]),
            "File": view.file,
            "Group": view.group.map(lambda g: tr.GROUPS[g][1]),
            "Main finding": view.top_finding,
            "Confidence": (view.confidence * 100).round(0),
            "Next most likely": [f"{a} ({b:.0%})" if a else "" for a, b in zip(view.second_finding, view.second_confidence.fillna(0))],
            "Photo quality": view.quality_notes.replace("", "ok"),
        })
        st.caption("Select a row to see that photo in detail below the table. "
                   "Confidence = how sure the model is about the main finding.")
        event = st.dataframe(
            table, hide_index=True, use_container_width=True, height=min(38 * len(table) + 40, 560),
            on_select="rerun", selection_mode="single-row", key="batch_table",
            column_order=["#", "Photo", "File", "Group", "Main finding", "Confidence", "Next most likely", "Photo quality"],
            column_config={
                "#": st.column_config.NumberColumn(width=40),
                "Photo": st.column_config.ImageColumn(width=60),
                "File": st.column_config.TextColumn(width=210),
                "Group": st.column_config.TextColumn(width=140),
                "Main finding": st.column_config.TextColumn(width=110),
                "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%", width=120),
                "Next most likely": st.column_config.TextColumn(width=150),
                "Photo quality": st.column_config.TextColumn(width=150),
            })
        sel = event.selection.rows if event is not None and hasattr(event, "selection") else []
        if sel:
            r = view.iloc[sel[0]]
            _detail(r, res, config, model, device)

    # ---------------------------------------------------------------- patients
    if res["show_patients"]:
        with tabs[1]:
            pp = tr.per_patient(df)
            st.caption("One row per patient. The more urgent eye decides the patient's group. "
                       "Patient IDs are read from file names like 123_left.jpg.")
            st.dataframe(pd.DataFrame({
                "#": pp.priority, "Patient": pp.patient, "Photos": pp.photos,
                "Group": pp.group.map(lambda g: tr.GROUPS[g][1]),
                "Main finding": pp.main_finding, "Confidence": (pp.confidence * 100).round(0), "Eyes": pp.eyes}),
                hide_index=True, use_container_width=True,
                column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%", width=120)})

    # ---------------------------------------------------------------- downloads
    with tabs[-1]:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
        csv_df = df.drop(columns=[c for c in df.columns if c.startswith("p_") and c != "p_disease"])
        d1, d2, d3 = st.columns(3)
        d1.download_button("Results table (CSV, opens in Excel)", csv_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"triage_results_{stamp}.csv", mime="text/csv", use_container_width=True)
        d2.download_button("Printable report (HTML)", _report_html(res).encode("utf-8"),
                           file_name=f"triage_report_{stamp}.html", mime="text/html", use_container_width=True)
        slot = d3.empty()
        if "zip" not in res and slot.button("Prepare sorted photos (ZIP)", use_container_width=True):
            with st.spinner("Packing photos into group folders..."):
                res["zip"] = _sorted_zip(res)
        if "zip" in res:
            slot.download_button("Sorted photos (ZIP)", res["zip"], file_name=f"triage_sorted_{stamp}.zip",
                               mime="application/zip", use_container_width=True)
        st.caption("The ZIP has one folder per group (1_Refer, 2_Doctor_review, 3_Likely_normal, 4_Retake_photo), "
                   "photos numbered by urgency, plus the results table and the report. "
                   "Open the HTML report in a browser and use Print → Save as PDF for a PDF.")


def _detail(r, res, config, model, device):
    st.markdown("---")
    g = r.group
    st.markdown(f"#### Photo #{r.priority} · {html.escape(r.file)}")
    title = {"refer": f"{r.top_finding} — {r.confidence:.0%} confidence",
             "review": f"Model unsure — {r.top_finding} {r.confidence:.0%}" if r.top_finding else "Model unsure",
             "normal": f"Normal — {r.confidence:.0%} confidence",
             "retake": "Photo quality too poor"}[g]
    sub = f"Why: {html.escape(r.reason)}" + (f" · Photo quality: {html.escape(r.quality_notes)}" if r.quality_notes else "")
    ui.result_banner(g, title, sub)
    src = res["sources"][r.file]
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.image(tr._open(src), caption="Photo (colours corrected if needed)", use_container_width=True)
    with c2:
        if r.top_finding:
            probs = sorted(((c[2:], r[c]) for c in r.index if c.startswith("p_") and c != "p_disease"),
                           key=lambda t: t[1], reverse=True)
            st.markdown("**Probability for each class**")
            ui.prob_bars(probs)
            key = f"cam_{r.file}"
            if st.button(f"Show where the model looked for {r.top_finding}", key=key, icon=":material/visibility:",
                         use_container_width=True):
                with st.spinner("Making the heatmap..."):
                    ov, _ = gc.multiclass_cam(config, model, device, io.BytesIO(_src_bytes(src)), r.top_finding)
                st.session_state["batch_cam"] = (r.file, ov)
            cam = st.session_state.get("batch_cam")
            if cam and cam[0] == r.file:
                st.image(cam[1], caption=f"Grad-CAM: red/yellow = areas that pushed the model towards {r.top_finding}",
                         use_container_width=True)
        else:
            st.warning("This photo could not be analysed.")
