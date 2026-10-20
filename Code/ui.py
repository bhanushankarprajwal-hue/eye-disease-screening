"""
Shared look and building blocks for the web app (Code/app.py and Code/views/*).

One place for colours, spacing and the small components every page uses, so all pages look and behave
the same. Colours use transparent tints, so they work in both the light and the dark Streamlit theme.
"""
import html
import os
import tempfile

import streamlit as st

import predict_multiclass as pmc
import predict_myopia as pm
import triage as tr

PROJECT_ROOT = pmc.PROJECT_ROOT

# Status colours: the same four meanings everywhere in the app
STATUS = {
    "refer":  {"color": "#EF4444", "label": "Refer",          "icon": ":material/priority_high:"},
    "review": {"color": "#F59E0B", "label": "Doctor review",  "icon": ":material/help:"},
    "normal": {"color": "#22C55E", "label": "Likely normal",  "icon": ":material/check_circle:"},
    "retake": {"color": "#94A3B8", "label": "Retake photo",   "icon": ":material/photo_camera:"},
    "info":   {"color": "#3B82F6", "label": "",               "icon": ":material/info:"},
}

# ------------------------------------------------------------------ palette
NAVY, BLUE, TEAL = "#0B2545", "#1D4ED8", "#0E9F9A"

# Small line icons (inline SVG, stroke = currentColor) used in headers and cards
ICON = {
    "home": '<path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/>',
    "eye": '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3.2"/>',
    "scan": '<path d="M4 8V5a1 1 0 011-1h3"/><path d="M16 4h3a1 1 0 011 1v3"/><path d="M20 16v3a1 1 0 01-1 1h-3"/>'
            '<path d="M8 20H5a1 1 0 01-1-1v-3"/><circle cx="12" cy="12" r="4"/><path d="M12 8v1.5M12 14.5V16M8 12h1.5M14.5 12H16"/>',
    "glasses": '<circle cx="6.5" cy="14" r="3.5"/><circle cx="17.5" cy="14" r="3.5"/><path d="M10 14h4"/>'
               '<path d="M3 14l1.5-7H7"/><path d="M21 14l-1.5-7H17"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/><path d="M3 17.5l9 5 9-5" opacity=".55"/>',
    "chart": '<path d="M4 20V10"/><path d="M10 20V4"/><path d="M16 20v-7"/><path d="M22 20H2"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/><path d="M8.5 12l2.5 2.5 4.5-5"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 010 18M12 3a14 14 0 000 18"/>',
}


def icon(name, size=22, color="currentColor", stroke=1.8):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
            f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</svg>')


# Stylised retina (fundus) drawing for the home page banner
RETINA_SVG = """
<svg viewBox="0 0 220 220" width="200" height="200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="fund" cx="45%" cy="45%" r="60%">
      <stop offset="0%" stop-color="#F7A548"/><stop offset="55%" stop-color="#D9662B"/><stop offset="100%" stop-color="#8A2E12"/>
    </radialGradient>
    <radialGradient id="disc" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#FFF6C8"/><stop offset="70%" stop-color="#FBD46D"/><stop offset="100%" stop-color="#F0A93B"/>
    </radialGradient>
  </defs>
  <circle cx="110" cy="110" r="104" fill="none" stroke="rgba(255,255,255,0.35)" stroke-width="1.5" stroke-dasharray="4 6"/>
  <circle cx="110" cy="110" r="92" fill="url(#fund)"/>
  <g fill="none" stroke="#7A1E0E" stroke-linecap="round" opacity="0.85">
    <path d="M150 104 C 120 80, 90 70, 40 66" stroke-width="3"/>
    <path d="M150 104 C 118 125, 88 140, 42 150" stroke-width="3"/>
    <path d="M150 100 C 130 60, 120 40, 108 22" stroke-width="2.4"/>
    <path d="M152 112 C 135 150, 128 175, 118 198" stroke-width="2.4"/>
    <path d="M100 75 C 85 95, 70 100, 30 105" stroke-width="1.5"/>
    <path d="M96 142 C 80 125, 66 122, 30 122" stroke-width="1.5"/>
    <path d="M160 100 C 178 86, 190 84, 200 80" stroke-width="1.8"/>
    <path d="M160 114 C 178 128, 188 134, 198 140" stroke-width="1.8"/>
  </g>
  <circle cx="152" cy="106" r="16" fill="url(#disc)"/>
  <circle cx="84" cy="110" r="11" fill="#7A2410" opacity="0.55"/>
  <g stroke="#7CF3E9" stroke-width="2" fill="none" opacity="0.95">
    <path d="M28 40 h-10 v10"/><path d="M192 40 h10 v10"/><path d="M28 180 h-10 v-10"/><path d="M192 180 h10 v-10"/>
  </g>
  <line x1="14" y1="96" x2="206" y2="96" stroke="#7CF3E9" stroke-width="1.6" opacity="0.7"/>
</svg>
"""

CSS = """
<style>
/* ---------- page ---------- */
.stApp { background: radial-gradient(1200px 600px at 70% -10%, #13284A 0%, #0B1526 55%, #09111F 100%); }
.block-container { padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1100px; }
[data-testid="stHeader"] { background: transparent; }
h2 { font-size: 1.22rem !important; font-weight: 700 !important; color: #F1F5FB; }
h3 { font-size: 1.08rem !important; font-weight: 700 !important; color: #F1F5FB; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #071020 0%, #0C1A30 100%); border-right: 1px solid rgba(255,255,255,0.06); }
[data-testid="stSidebar"] * { color: #DCE6F4 !important; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 8px; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(124,243,233,0.16) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color: #FFFFFF !important; font-weight: 700; }
[data-testid="stSidebarNavSeparator"], [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
.side-info { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
             padding: 0.7rem 0.8rem; font-size: 0.8rem; line-height: 1.5; margin-top: 0.4rem; }
.side-info b { color: #FFFFFF !important; }

/* ---------- header banners ---------- */
.hero { position: relative; border-radius: 18px; padding: 2rem 2.2rem; margin-bottom: 1.6rem; color: #fff;
        background: linear-gradient(120deg, #10284D 0%, #1B3F7A 50%, #0F6E6B 100%); border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 14px 40px rgba(0,0,0,0.35); display: flex; align-items: center; gap: 1.5rem; overflow: hidden; }
.hero .txt { flex: 1; position: relative; z-index: 1; }
.hero .tag { display:inline-block; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
             background: rgba(255,255,255,0.16); padding: 0.25rem 0.65rem; border-radius: 999px; margin-bottom: 0.7rem; }
.hero h1 { color: #fff !important; font-size: 2.1rem !important; font-weight: 800 !important; margin: 0 0 0.5rem 0 !important;
           padding: 0 !important; line-height: 1.15; }
.hero p { color: rgba(255,255,255,0.88); font-size: 1.02rem; margin: 0; max-width: 620px; line-height: 1.5; }
.hero .art { flex-shrink: 0; position: relative; z-index: 1; }
.hero::after { content: ""; position: absolute; right: -80px; top: -80px; width: 320px; height: 320px; border-radius: 50%;
               background: radial-gradient(circle, rgba(124,243,233,0.25), rgba(124,243,233,0) 70%); }
@media (max-width: 760px) { .hero .art { display: none; } .hero { padding: 1.4rem; } }

.phead { display: flex; align-items: center; gap: 1rem; border-radius: 16px; padding: 1.15rem 1.4rem; margin-bottom: 1.4rem;
         background: linear-gradient(120deg, #10284D 0%, #1B3F7A 65%, #0F6E6B 130%); color: #fff;
         border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
.phead .ico { width: 52px; height: 52px; border-radius: 14px; background: rgba(255,255,255,0.14); display: flex;
              align-items: center; justify-content: center; flex-shrink: 0; }
.phead h1 { color: #fff !important; font-size: 1.55rem !important; font-weight: 800 !important; margin: 0 !important; padding: 0 !important; }
.phead p { color: rgba(255,255,255,0.85); margin: 0.15rem 0 0 0; font-size: 0.95rem; }

/* ---------- steps ---------- */
.step { display:flex; align-items:center; gap:0.6rem; margin: 1.5rem 0 0.6rem 0; font-weight: 700; font-size: 1.04rem; color: #F1F5FB; }
.step-n { display:inline-flex; align-items:center; justify-content:center; width:1.75rem; height:1.75rem; border-radius:50%;
          background: linear-gradient(135deg, #2F6BFF, #14B8A6); color:#fff; font-size:0.88rem; font-weight:800;
          box-shadow: 0 3px 10px rgba(20,184,166,0.3); }

/* ---------- cards ---------- */
.card { background: linear-gradient(180deg, #15253F, #111E34); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
        padding: 1.2rem 1.25rem 1rem 1.25rem; box-shadow: 0 8px 24px rgba(0,0,0,0.28); min-height: 11.5rem; position: relative; overflow: hidden; }
.card::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: var(--accent, #1D4ED8); }
.card .ci { width: 46px; height: 46px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
            margin-bottom: 0.75rem; }
.card h4 { margin: 0 0 0.35rem 0; font-size: 1.06rem; font-weight: 700; color: #F1F5FB; }
.card p { margin: 0; color: #A9B8CE; font-size: 0.9rem; line-height: 1.5; }
[data-testid="stPageLink"] a { background: #15253F; border: 1px solid rgba(255,255,255,0.12); border-radius: 10px; padding: 0.35rem 0.8rem;
                               margin-top: 0.45rem; }
[data-testid="stPageLink"] a:hover { border-color: #14B8A6; background: #1A2E4D; }
[data-testid="stPageLink"] a span { color: #5EEAD4 !important; font-weight: 600; }

.kpi { background: linear-gradient(180deg, #15253F, #111E34); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px;
       padding: 0.95rem 1.05rem; min-height: 6.4rem; box-shadow: 0 8px 24px rgba(0,0,0,0.25); display: flex; gap: 0.8rem; align-items: flex-start; margin-bottom: 0.8rem; }
.kpi .ki { width: 38px; height: 38px; border-radius: 10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.kpi .v { font-size: 1.55rem; font-weight: 800; line-height: 1.15; color: #F1F5FB; }
.kpi .l { font-size: 0.8rem; color: #9FB0C8; margin-top: 0.15rem; line-height: 1.35; }

.chips { display:flex; flex-wrap:wrap; gap:0.5rem; margin: 0.3rem 0 0.5rem 0; }
.chip { background:#15253F; border:1px solid rgba(255,255,255,0.1); border-radius:999px; padding:0.35rem 0.85rem; font-size:0.88rem;
        color:#E5ECF6; }
.chip i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:0.45rem; vertical-align: middle; }

/* ---------- inputs / widgets ---------- */
[data-testid="stFileUploaderDropzone"] { background: #13223A; border: 1.5px dashed #3E5F8F; border-radius: 12px; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: #14B8A6; background: #172A47; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #2F6BFF, #14B8A6) !important; border: 0 !important; color: #fff !important;
    font-weight: 700 !important; border-radius: 10px !important; box-shadow: 0 8px 20px rgba(20,184,166,0.25); padding: 0.6rem 1rem; }
.stButton > button[kind="primary"]:disabled, [data-testid="stBaseButton-primary"]:disabled {
    background: #22324D !important; box-shadow: none; color: #7F8FA8 !important; }
[data-testid="stExpander"] details { background: #13223A; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
[data-testid="stImage"] img { border-radius: 12px; }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.3rem; }
[data-testid="stTabs"] button[role="tab"] { background: #13223A; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px 10px 0 0;
                                            padding: 0.45rem 0.9rem; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background: #1A3157; border-color: rgba(94,234,212,0.35); }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p { color: #5EEAD4; font-weight: 700; }
.panel { background:#13223A; border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:1rem 1.2rem; }

/* ---------- result banner ---------- */
.result { border-radius: 14px; padding: 1.1rem 1.3rem; margin: 0.4rem 0 1rem 0; border: 1px solid; border-left-width: 6px;
          background: #13223A; box-shadow: 0 10px 26px rgba(0,0,0,0.3); }
.result .t { font-size: 1.4rem; font-weight: 800; line-height: 1.25; color: #F8FAFC; }
.result .s { font-size: 0.96rem; margin-top: 0.3rem; color: #C7D2E3; }
.pill { display:inline-block; padding: 0.14rem 0.65rem; border-radius: 999px; font-size: 0.76rem; font-weight: 700;
        color: #fff; vertical-align: middle; letter-spacing: 0.02em; }

/* ---------- probability bars ---------- */
.bar-row { display:flex; align-items:center; gap:0.7rem; margin: 0.38rem 0; font-size: 0.93rem; color: #DCE6F4; }
.bar-name { width: 112px; flex-shrink: 0; }
.bar-track { flex: 1; background: rgba(255,255,255,0.08); border-radius: 6px; height: 11px; overflow: hidden; display:flex; }
.bar-fill { height: 100%; background: #3B5A8C; border-radius: 6px; }
.bar-fill.top { background: linear-gradient(90deg, #2F6BFF, #14B8A6); }
.bar-val { width: 54px; text-align: right; font-variant-numeric: tabular-nums; }
.bar-row.top .bar-name, .bar-row.top .bar-val { font-weight: 800; color: #FFFFFF; }

/* ---------- footer ---------- */
.foot { font-size: 0.8rem; color: #8FA0B8; margin-top: 2.5rem; padding: 0.9rem 1rem; border-radius: 12px;
        background: #0F1B2E; border: 1px solid rgba(255,255,255,0.06); display: flex; gap: 0.6rem; align-items: flex-start; }
</style>
"""

DISCLAIMER = ("Research prototype from a final-year capstone project. Trained on the public ODIR-5K and RFMiD datasets. "
              "It is a screening aid, not a diagnosis, and it is not validated for real patient use.")


def apply_style():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(tag, title, subtitle):
    """Large banner with the retina drawing (home page)."""
    st.markdown(f'<div class="hero"><div class="txt"><div class="tag">{html.escape(tag)}</div><h1>{html.escape(title)}</h1>'
                f'<p>{subtitle}</p></div><div class="art">{RETINA_SVG}</div></div>', unsafe_allow_html=True)


def page_header(title, subtitle, icon_name="eye"):
    st.markdown(f'<div class="phead"><div class="ico">{icon(icon_name, 28, "#FFFFFF", 1.9)}</div>'
                f'<div><h1>{html.escape(title)}</h1><p>{subtitle}</p></div></div>', unsafe_allow_html=True)


def step(n, text):
    st.markdown(f'<div class="step"><span class="step-n">{n}</span>{html.escape(text)}</div>', unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="foot">{icon("shield", 18, "#8FA0B8")}<span>{DISCLAIMER}</span></div>', unsafe_allow_html=True)


def tint(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def result_banner(status, title, subtitle, pill_text=None):
    c = STATUS[status]["color"]
    label = STATUS[status]["label"] if pill_text is None else pill_text
    pill = f'<span class="pill" style="background:{c}">{html.escape(label)}</span>&nbsp; ' if label else ""
    st.markdown(
        f'<div class="result" style="background:linear-gradient(90deg,{tint(c, 0.18)},#13223A 70%);'
        f'border-color:{tint(c, 0.30)};border-left-color:{c}">'
        f'<div class="t">{pill}{html.escape(title)}</div><div class="s">{subtitle}</div></div>',
        unsafe_allow_html=True)


def prob_bars(ranked):
    """ranked: list of (name, probability), highest first."""
    rows = []
    for i, (n, p) in enumerate(ranked):
        top = " top" if i == 0 else ""
        rows.append(f'<div class="bar-row{top}"><div class="bar-name">{html.escape(n)}</div>'
                    f'<div class="bar-track"><div class="bar-fill{top}" style="width:{p*100:.1f}%"></div></div>'
                    f'<div class="bar-val">{p*100:.1f}%</div></div>')
    st.markdown('<div class="panel">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def card(title, text, icon_name="eye", accent="#1D4ED8"):
    st.markdown(f'<div class="card" style="--accent:{accent}"><div class="ci" style="background:{tint(accent, 0.12)}">'
                f'{icon(icon_name, 24, accent)}</div><h4>{html.escape(title)}</h4><p>{text}</p></div>',
                unsafe_allow_html=True)


def kpi(value, label, icon_name="chart", accent="#1D4ED8"):
    st.markdown(f'<div class="kpi"><div class="ki" style="background:{tint(accent, 0.12)}">{icon(icon_name, 20, accent)}</div>'
                f'<div><div class="v">{html.escape(value)}</div><div class="l">{html.escape(label)}</div></div></div>',
                unsafe_allow_html=True)


def chips(items):
    """items: list of (text, colour)."""
    st.markdown('<div class="chips">' + "".join(f'<span class="chip"><i style="background:{c}"></i>{html.escape(t)}</span>'
                                                for t, c in items) + "</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ models (loaded once, shared by all pages)
@st.cache_resource(show_spinner="Loading the 8-disease model (first time only)...")
def multiclass_model():
    return pmc.load_multiclass_model()


@st.cache_resource(show_spinner="Loading the Myopia model (first time only)...")
def myopia_model():
    return pm.load_final_model()


def load_or_stop(loader):
    try:
        return loader()
    except Exception as e:  # noqa: BLE001 -- shown to the user
        st.error(f"The model could not be loaded: {e}")
        st.stop()


def save_uploads(files, tmpdir):
    paths = []
    for i, f in enumerate(files):
        p = os.path.join(tmpdir, f"{i}_{os.path.basename(f.name)}")
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        paths.append(p)
    return paths


def sample_images():
    """Held-out test photos shipped with the project (test_images/), for quick demos."""
    d = os.path.join(PROJECT_ROOT, "test_images")
    if not os.path.isdir(d):
        return {}
    out = {}
    for f in sorted(os.listdir(d)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            parts = os.path.splitext(f)[0].split("_")
            label = f"{parts[1]} — photo {'_'.join(parts[2:])}" if len(parts) >= 3 else f
            out[label] = os.path.join(d, f)
    return out


def triage_status(ranked):
    """Same rule as the batch tool (Code/triage.py), so one photo and many photos get the same answer."""
    limits, _ = tr.load_thresholds()
    names = [n for n, _ in ranked]
    probs = dict(ranked)
    top, top_p = ranked[0]
    if top != "Normal" and top_p >= limits["refer_min_confidence"]:
        return "refer", limits
    if top == "Normal" and probs["Normal"] >= limits["normal_min_confidence"]:
        return "normal", limits
    return "review", limits
