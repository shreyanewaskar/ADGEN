# -*- coding: utf-8 -*-
"""
AdGen AI — Streamlit UI
Wires the postermaking.py pipeline (SD-Turbo + fine-tuned Transformer)
into the premium lavender UI.
"""

import os
import sys
import io
import time

import streamlit as st
from PIL import Image

# ── Make sure postermaking.py (same folder) is importable ──────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ─── Basic content safety (prompt-level) ─────────────────────────────────────
# NOTE: this is a lightweight keyword check on user-provided TEXT only. It is
# not a substitute for an image-level NSFW classifier — the local SD-Turbo
# model folder has no safety_checker component saved, so generated images
# are not moderated. For a public/production deployment, layer a proper
# image-level safety checker on top of this.
UNSAFE_KEYWORDS = [
    "nude", "naked", "nsfw", "porn", "pornographic", "sexual intercourse",
    "explicit sexual", "gore", "graphic violence", "murder", "suicide",
    "self-harm", "self harm", "firearm", "assault rifle", "cocaine",
    "heroin", "meth ", "nazi", "terrorist attack",
]

MAX_GENERATIONS_PER_SESSION = 15


def check_content_safety(*texts) -> str | None:
    """Returns a warning message if any text contains a disallowed term,
    else None. Basic keyword safeguard — see note above."""
    blob = " ".join(t or "" for t in texts).lower()
    for kw in UNSAFE_KEYWORDS:
        if kw in blob:
            return (
                "Your input contains content that isn't allowed here. "
                "Please revise your product description/prompt and try again."
            )
    return None


def strip_exif(pil_img: Image.Image) -> Image.Image:
    """Rebuild the image from raw pixels, dropping EXIF/metadata (may
    contain GPS/device info) from uploaded photos."""
    import numpy as np
    arr = np.asarray(pil_img.convert("RGB"))
    return Image.fromarray(arr)


def check_rate_limit() -> str | None:
    """Very basic per-session generation cap (resets on page refresh — not
    a robust server-side limiter, but a reasonable default without adding
    an auth/backend layer)."""
    if st.session_state.get("generation_count", 0) >= MAX_GENERATIONS_PER_SESSION:
        return (
            f"You've reached this session's limit of {MAX_GENERATIONS_PER_SESSION} "
            "generations. Refresh the page to start a new session."
        )
    return None

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdGen AI – Advertisement Poster Creator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #FAFAFF 45%, #F3F1FF 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #303044 !important;
    min-height: 100vh;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stAppViewContainer"] > section > div { padding-top: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

body::before {
    content: ''; position: fixed; top: -200px; left: -200px;
    width: 700px; height: 700px;
    background: radial-gradient(circle, rgba(113,103,217,0.13) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none; z-index: 0;
}
body::after {
    content: ''; position: fixed; bottom: -150px; right: -150px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(113,103,217,0.10) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none; z-index: 0;
}

/* ── Navbar ── */
.navbar {
    position: sticky; top: 0; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 48px;
    background: rgba(255,255,255,0.78);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(230,228,242,0.6);
}
.navbar-brand { font-size: 18px; font-weight: 600; color: #303044; display: flex; align-items: center; gap: 6px; }
.navbar-brand span { color: #7167D9; }
.navbar-links { display: flex; gap: 32px; }
.navbar-links a { font-size: 13px; font-weight: 500; color: #77778A; text-decoration: none; transition: color 0.2s; }
.navbar-links a:hover, .navbar-links a.active { color: #7167D9; }
.nav-account {
    font-size: 13px; font-weight: 500; padding: 8px 20px;
    border: 1.5px solid #E6E4F2; border-radius: 50px;
    color: #303044; background: white; cursor: pointer; transition: all 0.2s;
}
.nav-account:hover { border-color: #7167D9; color: #7167D9; }

/* ── Hero ── */
.hero-section { text-align: center; padding: 72px 24px 36px; position: relative; z-index: 1; }
.hero-title { font-size: 56px; font-weight: 500; line-height: 1.18; color: #303044; letter-spacing: -1.5px; margin-bottom: 18px; }
.hero-title .purple { color: #7167D9; }
.hero-sub { font-size: 16px; font-weight: 400; color: #77778A; max-width: 480px; margin: 0 auto 22px; line-height: 1.65; }
.ai-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: #E8E5FF; color: #7167D9;
    font-size: 12px; font-weight: 600; letter-spacing: 0.3px;
    padding: 7px 18px; border-radius: 50px; border: 1px solid rgba(113,103,217,0.25);
}

/* ── Float cards ── */
.float-card {
    background: rgba(255,255,255,0.84); backdrop-filter: blur(12px);
    border: 1px solid #E6E4F2; border-radius: 20px; padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(113,103,217,0.08); margin-bottom: 14px;
}
.float-card-icon { font-size: 18px; margin-bottom: 6px; color: #7167D9; }
.float-card-title { font-size: 13px; font-weight: 600; color: #303044; }
.float-card-sub { font-size: 11px; color: #77778A; margin-top: 2px; }

/* ── Mode toggle (visual pill indicator) ── */
.mode-pill-wrap {
    display: flex; gap: 0; background: #F0EEFF; border-radius: 50px;
    padding: 4px; width: fit-content; border: 1px solid #E0DCFF; margin-bottom: 6px;
}

/* ── Main card ── */
.main-card {
    background: rgba(255,255,255,0.84); backdrop-filter: blur(20px);
    border: 1px solid #E6E4F2; border-radius: 28px; padding: 44px 52px;
    box-shadow: 0 8px 48px rgba(113,103,217,0.10), 0 2px 8px rgba(113,103,217,0.06);
    max-width: 1060px; margin: 0 auto; position: relative;
}
.main-card-title { font-size: 22px; font-weight: 600; color: #303044; margin-bottom: 6px; }
.main-card-sub   { font-size: 14px; color: #77778A; margin-bottom: 32px; }

/* ── Section label ── */
.section-label { font-size: 13px; font-weight: 600; color: #303044; margin-bottom: 8px; letter-spacing: 0.1px; }

/* ── Upload box ── */
.upload-box {
    background: #F8F7FF; border: 2px dashed #C5C0F0; border-radius: 20px;
    padding: 40px 24px; text-align: center; cursor: pointer; min-height: 220px;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px;
    transition: all 0.2s;
}
.upload-box:hover { border-color: #7167D9; background: #F0EEFF; }
.upload-icon  { font-size: 32px; color: #B0A8E8; }
.upload-label { font-size: 15px; font-weight: 500; color: #77778A; }
.upload-hint  { font-size: 12px; color: #B0AEC8; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important; color: #303044 !important;
    background: #FAFAFF !important; border: 1.5px solid #E6E4F2 !important;
    border-radius: 14px !important; padding: 12px 16px !important;
    box-shadow: none !important; transition: border 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7167D9 !important;
    box-shadow: 0 0 0 3px rgba(113,103,217,0.10) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important; color: #303044 !important;
}
.stSelectbox > div > div {
    border-radius: 14px !important; border: 1.5px solid #E6E4F2 !important; background: #FAFAFF !important;
}

/* ── Soft divider ── */
.soft-divider { height: 1px; background: #ECEAF6; margin: 28px 0; border-radius: 1px; }

/* ── Streamlit buttons ── */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #7167D9 !important; color: white !important;
    border: none !important; border-radius: 50px !important;
    font-size: 15px !important; font-weight: 600 !important;
    padding: 13px 40px !important; width: 100% !important;
    box-shadow: 0 6px 24px rgba(113,103,217,0.28) !important; transition: all 0.22s !important;
}
.stButton > button:hover {
    background: #6258CC !important;
    box-shadow: 0 10px 32px rgba(113,103,217,0.40) !important;
    transform: translateY(-1px) !important;
}

/* Status pills — override only those two buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    font-size: 13px !important; padding: 9px 22px !important;
    border-radius: 50px !important; width: auto !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #F8F7FF !important; border: 2px dashed #C5C0F0 !important;
    border-radius: 20px !important; padding: 8px !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }

/* ── Poster result card ── */
.result-card {
    background: rgba(255,255,255,0.90); border: 1.5px solid #D8D4F5;
    border-radius: 24px; padding: 32px 36px; margin-top: 32px;
    box-shadow: 0 8px 40px rgba(113,103,217,0.13);
}
.result-title { font-size: 18px; font-weight: 700; color: #303044; margin-bottom: 4px; }
.result-sub   { font-size: 13px; color: #77778A; margin-bottom: 20px; }

/* ── Ad copy box ── */
.copy-box {
    background: #F3F1FF; border: 1px solid #DDD8F5; border-radius: 16px;
    padding: 20px 24px; margin-top: 16px;
}
.copy-row { margin-bottom: 12px; }
.copy-label { font-size: 11px; font-weight: 700; color: #9B96D4; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 3px; }
.copy-value { font-size: 14px; color: #303044; line-height: 1.5; }

/* ── AI pipeline ── */
.pipeline-wrap {
    display: flex; align-items: center; justify-content: center;
    flex-wrap: wrap; margin-top: 32px; padding-top: 28px;
    border-top: 1px solid #ECEAF6; gap: 0;
}
.pipe-step { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 0 18px; }
.pipe-icon {
    width: 44px; height: 44px; border-radius: 14px;
    background: #F0EEFF; border: 1px solid #E0DCFF;
    display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.pipe-label { font-size: 11px; font-weight: 500; color: #77778A; text-align: center; }
.pipe-arrow { font-size: 16px; color: #C5C0F0; padding-bottom: 18px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #D5D0F0; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Model loader (cached — runs once per session) ──────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Import postermaking — this triggers SD-Turbo + Transformer to load onto GPU."""
    import postermaking as pm
    return pm


def run_pipeline(title: str, category: str, description: str):
    """Run the full text → poster pipeline."""
    pm = load_pipeline()
    return pm.text_to_poster(title, category, description)


def run_restyle(image: Image.Image, prompt: str, strength: float):
    """Image-to-image restyle — keeps the uploaded product recognizable
    while the prompt steers background/style. Reuses the same loaded
    SD-Turbo pipeline as run_pipeline (no extra model load)."""
    pm = load_pipeline()
    return pm.restyle_image(image, prompt, strength=strength)


# ─── About-page model loaders (independent of postermaking.py's pipeline) ───
@st.cache_resource(show_spinner=False)
def load_ae_model():
    """Load the trained Keras autoencoder from Model/autoencoder_full_1.keras."""
    import keras

    ae_path = os.path.abspath(os.path.join(SRC_DIR, "..", "Model", "autoencoder_full_1.keras"))
    return keras.models.load_model(ae_path, compile=False)


@st.cache_resource(show_spinner=False)
def load_vae_model():
    """Load the trained VAE from Model/vae_best."""
    from diffusers import AutoencoderKL
    vae_dir = os.path.abspath(os.path.join(SRC_DIR, "..", "Model", "vae_best"))
    vae = AutoencoderKL.from_pretrained(vae_dir)
    vae.eval()
    return vae


def run_ae_reconstruction(pil_img: Image.Image) -> Image.Image:
    import numpy as np
    model = load_ae_model()
    img = pil_img.convert("RGB").resize((128, 128))
    batch = (np.asarray(img).astype("float32") / 255.0)[None, ...]
    out = model.predict(batch, verbose=0)[0]
    out = (np.clip(out, 0, 1) * 255).astype("uint8")
    return Image.fromarray(out)


def compute_ae_reconstruction_error(pil_img: Image.Image) -> float:
    """AE-based upload quality gate: how well the AE reconstructs this image.
    A high MSE means the image is likely blurry, corrupted, or off-domain
    (not a clean product-on-plain-background shot)."""
    import numpy as np
    model = load_ae_model()
    img = pil_img.convert("RGB").resize((128, 128))
    batch = (np.asarray(img).astype("float32") / 255.0)[None, ...]
    recon = model.predict(batch, verbose=0)[0]
    return float(np.mean((batch[0] - recon) ** 2))


AE_QUALITY_MSE_THRESHOLD = 0.01  # ~16x the AE's measured test-set MSE (0.00062)


def run_vae_reconstruction(pil_img: Image.Image) -> Image.Image:
    import torch
    import numpy as np
    vae = load_vae_model()
    img = pil_img.convert("RGB").resize((256, 256))
    arr = (np.asarray(img).astype("float32") / 127.5) - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        latent  = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
        decoded = vae.decode(latent / vae.config.scaling_factor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    out = (decoded[0].permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(out)


# ─── Session State ───────────────────────────────────────────────────────────
defaults = {
    "ad_status":        "New",
    "input_mode":       "custom",   # "custom" | "text" | "restyle"
    "poster_result":    None,
    "restyle_result":   None,
    "page":             "create",   # "create" | "about"
    "generation_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Text-prompt mode pre-filled example values ──────────────────────────────
DEMO_TITLE       = "Modern Ceramic Table Lamp"
DEMO_DESCRIPTION = (
    "A modern ceramic table lamp shown alone with no human or body parts. "
    "The lamp has a smooth rounded ceramic base, approximately 30 cm in height, "
    "a minimalist cylindrical fabric shade, a warm white light, and a clean "
    "contemporary design. The main color is matte white with subtle beige "
    "detailing. The lamp is positioned upright on a simple light wooden surface, "
    "clearly showing the base, shade, and overall shape. Place the lamp in a "
    "cozy light gray studio-style Home & Garden setting with soft realistic "
    "shadows, warm professional lighting, and no other objects."
)
DEMO_CATEGORY    = "Home & Garden"

# ─── Navbar ──────────────────────────────────────────────────────────────────
# Render the glass navbar with real Streamlit buttons for Create and About
_page = st.session_state.page

st.markdown(f"""
<div class="navbar">
  <div class="navbar-brand"><span>✦</span> AdGen AI</div>
  <div class="navbar-links" id="nav-links-placeholder"></div>
  <button class="nav-account">My Account</button>
</div>
<style>
  /* hide default button chrome inside navbar cols */
  [data-testid="stHorizontalBlock"] {{ position:relative; }}
</style>
""", unsafe_allow_html=True)

# Nav buttons rendered just below the navbar HTML so they visually overlay it
st.markdown("""
<style>
.st-key-navlinks_row [data-testid="stHorizontalBlock"] { gap:10px !important; }
.st-key-navlinks_row [data-testid="column"]:not(:first-child) {
    flex:0 0 auto !important; width:auto !important; min-width:0 !important;
}
</style>
""", unsafe_allow_html=True)

navlinks_container = st.container(key="navlinks_row")
with navlinks_container:
    nav1, nav2, nav3, nav4, nav5 = st.columns([2, 1, 1, 1, 1])
with nav2:
    create_style = "color:#7167D9;font-weight:700;" if _page == "create" else ""
    st.markdown(
        f'<style>.st-key-nav_create button{{background:transparent!important;'
        f'box-shadow:none!important;color:{"#7167D9" if _page=="create" else "#77778A"}!important;'
        f'font-size:13px!important;padding:4px 0!important;border:none!important;'
        f'font-weight:{"700" if _page=="create" else "500"}!important;width:auto!important;}}</style>',
        unsafe_allow_html=True
    )
    with st.container(key="nav_create"):
        if st.button("Create", key="nav_create_btn"):
            st.session_state.page = "create"
            st.rerun()
with nav3:
    st.markdown(
        '<style>.st-key-nav_posters button{background:transparent!important;'
        'box-shadow:none!important;color:#77778A!important;'
        'font-size:13px!important;padding:4px 0!important;border:none!important;'
        'font-weight:500!important;width:auto!important;}</style>',
        unsafe_allow_html=True
    )
    with st.container(key="nav_posters"):
        st.button("My Posters", key="nav_posters_btn", disabled=True)
with nav4:
    st.markdown(
        '<style>.st-key-nav_templates button{background:transparent!important;'
        'box-shadow:none!important;color:#77778A!important;'
        'font-size:13px!important;padding:4px 0!important;border:none!important;'
        'font-weight:500!important;width:auto!important;}</style>',
        unsafe_allow_html=True
    )
    with st.container(key="nav_templates"):
        st.button("Templates", key="nav_templates_btn", disabled=True)
with nav5:
    st.markdown(
        f'<style>.st-key-nav_about button{{background:transparent!important;'
        f'box-shadow:none!important;color:{"#7167D9" if _page=="about" else "#77778A"}!important;'
        f'font-size:13px!important;padding:4px 0!important;border:none!important;'
        f'font-weight:{"700" if _page=="about" else "500"}!important;width:auto!important;}}</style>',
        unsafe_allow_html=True
    )
    with st.container(key="nav_about"):
        if st.button("About", key="nav_about_btn"):
            st.session_state.page = "about"
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# ── CREATE PAGE ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "create":

 # ─── Hero ─────────────────────────────────────────────────────────────────────
 st.markdown("""
 <div class="hero-section">
   <div class="hero-title">
     Create your perfect<br>
     <span class="purple">Advertisement Poster.</span>
   </div>
   <div class="hero-sub">
     Turn your product details into a professional advertisement with AI.
   </div>
   <div class="ai-badge">✦ Powered by AI</div>
 </div>
 """, unsafe_allow_html=True)

 # ─── Three-column layout: float-left | main | float-right ────────────────────
 col_left, col_main, col_right = st.columns([1, 6, 1])

 # ── Left float card ──
 with col_left:
     st.markdown("""
     <div style="padding-top:44px;">
       <div class="float-card">
         <div class="float-card-icon">✦</div>
         <div class="float-card-title">AI Copy</div>
         <div class="float-card-sub">Headline &amp; tagline<br>generated</div>
       </div>
     </div>
     """, unsafe_allow_html=True)

 # ── Right float cards ──
 with col_right:
     st.markdown("""
     <div style="height:100%;display:flex;flex-direction:column;justify-content:flex-end;padding-bottom:24px;">
       <div class="float-card">
         <div class="float-card-icon">◇</div>
         <div class="float-card-title">AI Visuals</div>
         <div class="float-card-sub">Background<br>generated</div>
       </div>
       <div class="float-card" style="margin-top:12px;">
         <div class="float-card-icon">✧</div>
         <div class="float-card-title">Smart Layout</div>
         <div class="float-card-sub">Multiple designs<br>generated</div>
       </div>
     </div>
     """, unsafe_allow_html=True)

 # ── Main column ──
 with col_main:

    st.markdown("""
    <div class="main-card">
      <div class="main-card-title">Create your Advertisement</div>
      <div class="main-card-sub">Tell us about your product and let AI design the advertisement.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input mode toggle ────────────────────────────────────────────────────
    st.markdown('<div style="margin:20px 0 8px;">', unsafe_allow_html=True)

    # Single pill toggle: real Streamlit buttons, styled to look like pill segments.
    custom_active  = st.session_state.input_mode == "custom"
    text_active    = st.session_state.input_mode == "text"
    restyle_active = st.session_state.input_mode == "restyle"

    def _pill_style(active: bool) -> str:
        bg     = "white" if active else "transparent"
        color  = "#7167D9" if active else "#9B96D4"
        shadow = "0 2px 10px rgba(113,103,217,0.15)" if active else "none"
        return f"""
        background:{bg} !important; color:{color} !important;
        box-shadow:{shadow} !important; border:none !important;
        border-radius:50px !important; padding:9px 28px !important;
        font-size:13px !important; font-weight:600 !important;
        width:auto !important; transition:all 0.2s !important;
        """

    st.markdown(f"""
    <style>
    .st-key-toggle_row [data-testid="stHorizontalBlock"] {{
        display:flex !important; gap:0 !important;
        background:#F0EEFF; border-radius:50px; padding:4px;
        width:fit-content !important; margin:0 auto 6px !important;
        border:1px solid #E0DCFF;
    }}
    .st-key-toggle_row [data-testid="column"] {{
        width:auto !important; flex:0 0 auto !important; min-width:0 !important;
    }}
    .st-key-toggle_custom button  {{ {_pill_style(custom_active)} }}
    .st-key-toggle_text button    {{ {_pill_style(text_active)} }}
    .st-key-toggle_restyle button {{ {_pill_style(restyle_active)} }}
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="toggle_row"):
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            if st.button("✎  Custom Input", key="toggle_custom"):
                st.session_state.input_mode = "custom"
                st.session_state.poster_result = None
                st.rerun()
        with tc2:
            if st.button("✦  Text Prompt", key="toggle_text"):
                st.session_state.input_mode = "text"
                st.session_state.poster_result = None
                st.rerun()
        with tc3:
            if st.button("🖼  Restyle Photo", key="toggle_restyle"):
                st.session_state.input_mode = "restyle"
                st.session_state.poster_result = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── TEXT PROMPT MODE ─────────────────────────────────────────────────────
    if st.session_state.input_mode == "text":

        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <div style="width:36px;height:36px;background:#EDE9FF;border-radius:12px;
                      display:flex;align-items:center;justify-content:center;
                      font-size:16px;flex-shrink:0;">✦</div>
          <div>
            <div style="font-size:16px;font-weight:700;color:#303044;">Text Prompt</div>
            <div style="font-size:13px;color:#77778A;margin-top:1px;">
              Enter your product details as text — AI handles everything else.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        st.text_input(
            "Product Title",
            placeholder=f"e.g. {DEMO_TITLE}",
            key="tp_title"
        )
        st.text_area(
            "Product Description",
            placeholder=DEMO_DESCRIPTION,
            height=160,
            key="tp_desc"
        )

        CATEGORIES = ["Select category", "Home & Garden", "Sporting Goods",
                      "Electronics", "Health & Beauty", "Toys & Games"]
        st.selectbox("Category", CATEGORIES, key="tp_category")

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── CUSTOM INPUT MODE ────────────────────────────────────────────────────
    if st.session_state.input_mode == "custom":
        img_col, info_col = st.columns([1, 1.4], gap="large")

        with img_col:
            st.markdown('<div class="section-label">Product Image</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Product Image (optional)",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="product_image"
            )
            if uploaded:
                pil_img = strip_exif(Image.open(uploaded))
                st.image(pil_img, use_container_width=True)
                st.markdown(
                    '<div style="text-align:center;margin-top:6px;">'
                    '<span style="font-size:12px;color:#7167D9;font-weight:500;">⟳ Change Image</span>'
                    '</div>', unsafe_allow_html=True
                )

                # ── AE-based upload quality gate ─────────────────────────────
                cache_key = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
                if st.session_state.get("_ae_check_key") != cache_key:
                    try:
                        st.session_state["_ae_check_mse"] = compute_ae_reconstruction_error(pil_img)
                    except Exception:
                        st.session_state["_ae_check_mse"] = None
                    st.session_state["_ae_check_key"] = cache_key

                ae_mse = st.session_state.get("_ae_check_mse")
                if ae_mse is not None:
                    if ae_mse > AE_QUALITY_MSE_THRESHOLD:
                        st.warning(
                            f"⚠ Quality check: this image reconstructs poorly "
                            f"(error {ae_mse:.4f}, expected < {AE_QUALITY_MSE_THRESHOLD}). "
                            "It may be blurry, low-resolution, or not a clean product-on-plain-"
                            "background photo. You can still generate, but a clearer image "
                            "usually gives a better poster."
                        )
                    else:
                        st.markdown(
                            f'<div style="text-align:center;margin-top:4px;font-size:11px;color:#6FAE6F;">'
                            f'✓ Quality check passed (error {ae_mse:.4f})</div>',
                            unsafe_allow_html=True
                        )
            else:
                st.markdown("""
                <div class="upload-box">
                  <div class="upload-icon">↑</div>
                  <div class="upload-label">Upload Product Image</div>
                  <div class="upload-hint">Drag &amp; drop or browse<br>
                  <span style="font-size:11px;color:#C0BCE0;">(optional — AI generates one if omitted)</span></div>
                </div>
                """, unsafe_allow_html=True)

        with info_col:
            st.text_input("Product Title", placeholder="Enter your product name", key="product_title")
            st.text_area(
                "Product Description",
                placeholder="Describe your product, its features and benefits...",
                height=110, key="product_desc"
            )
            cat_col, brand_col = st.columns(2, gap="medium")
            with cat_col:
                st.selectbox(
                    "Category",
                    ["Select category", "Home & Garden", "Sporting Goods",
                     "Electronics", "Health & Beauty", "Toys & Games"],
                    key="product_category"
                )
            with brand_col:
                st.text_input("Brand", placeholder="Enter brand name", key="product_brand")

            st.markdown('<div class="section-label" style="margin-top:10px;">Product Status</div>',
                        unsafe_allow_html=True)
            s1, s2, _ = st.columns([1, 1.3, 2])
            with s1:
                label = "✓ New" if st.session_state.ad_status == "New" else "New"
                if st.button(label, key="status_new"):
                    st.session_state.ad_status = "New"; st.rerun()
            with s2:
                label = "✓ Second Hand" if st.session_state.ad_status == "Second Hand" else "Second Hand"
                if st.button(label, key="status_sh"):
                    st.session_state.ad_status = "Second Hand"; st.rerun()

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # ── RESTYLE MODE (image-to-image) ────────────────────────────────────────
    if st.session_state.input_mode == "restyle":

        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
          <div style="width:36px;height:36px;background:#EDE9FF;border-radius:12px;
                      display:flex;align-items:center;justify-content:center;
                      font-size:16px;flex-shrink:0;">🖼</div>
          <div>
            <div style="font-size:16px;font-weight:700;color:#303044;">Restyle Photo</div>
            <div style="font-size:13px;color:#77778A;margin-top:1px;">
              Upload your exact product photo and describe how to restyle it —
              the product stays recognizable, only the style/background changes.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

        restyle_img_col, restyle_info_col = st.columns([1, 1.4], gap="large")

        with restyle_img_col:
            st.markdown('<div class="section-label">Product Photo</div>', unsafe_allow_html=True)
            restyle_upload = st.file_uploader(
                "Product Photo",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key="restyle_image"
            )
            if restyle_upload:
                restyle_src_img = strip_exif(Image.open(restyle_upload))
                st.image(restyle_src_img, use_container_width=True)
            else:
                restyle_src_img = None
                st.markdown("""
                <div class="upload-box">
                  <div class="upload-icon">↑</div>
                  <div class="upload-label">Upload Product Photo</div>
                  <div class="upload-hint">Drag &amp; drop or browse<br>
                  <span style="font-size:11px;color:#C0BCE0;">(required for restyling)</span></div>
                </div>
                """, unsafe_allow_html=True)

        with restyle_info_col:
            st.text_area(
                "Restyle Prompt",
                placeholder="e.g. restyle the background to a clean white studio\ne.g. make it look outdoor and lifestyle",
                height=110,
                key="restyle_prompt"
            )
            restyle_strength = st.slider(
                "Restyle Strength",
                min_value=0.20, max_value=0.60, value=0.35, step=0.05,
                key="restyle_strength",
                help="Lower = keeps the product almost identical, only background/lighting shifts. "
                     "Higher = more creative freedom, but the product itself may start to change too."
            )

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        rgen_l, rgen_mid, rgen_r = st.columns([1, 2, 1])
        with rgen_mid:
            clicked_restyle = st.button("🎨  Restyle Image", key="generate_restyle")

        st.markdown(
            '<div style="text-align:center;margin-top:10px;font-size:13px;color:#77778A;">'
            'The product stays recognizable — only style/background changes.</div>',
            unsafe_allow_html=True
        )

        if clicked_restyle:
            if not restyle_src_img:
                st.warning("Please upload a product photo to restyle.")
                st.stop()
            restyle_prompt_val = st.session_state.get("restyle_prompt", "").strip()
            if not restyle_prompt_val:
                st.warning("Please describe how you'd like to restyle the image.")
                st.stop()

            safety_msg = check_content_safety(restyle_prompt_val)
            if safety_msg:
                st.warning(safety_msg)
                st.stop()

            rate_msg = check_rate_limit()
            if rate_msg:
                st.warning(rate_msg)
                st.stop()

            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.markdown("""
                <div style="text-align:center;padding:32px 0 20px;">
                  <div style="font-size:28px;margin-bottom:12px;">🎨</div>
                  <div style="font-size:16px;font-weight:600;color:#7167D9;margin-bottom:6px;">
                    AI is restyling your photo…
                  </div>
                  <div style="font-size:13px;color:#77778A;">
                    Encoding image · Applying style · Decoding result
                  </div>
                </div>
                """, unsafe_allow_html=True)

            try:
                restyled_img, prompt_used = run_restyle(
                    restyle_src_img, restyle_prompt_val, strength=restyle_strength
                )
                st.session_state.restyle_result = {
                    "original": restyle_src_img,
                    "restyled": restyled_img,
                    "prompt":   prompt_used,
                }
                st.session_state.generation_count += 1
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Restyling failed: {e}")
                st.stop()

            progress_placeholder.empty()

        if st.session_state.restyle_result:
            rres = st.session_state.restyle_result
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="result-title">✦ Restyled Photo</div>'
                f'<div class="result-sub">{rres["prompt"]}</div>',
                unsafe_allow_html=True
            )

            orig_col, restyled_col = st.columns(2, gap="large")
            with orig_col:
                st.markdown('<div class="section-label" style="text-align:center;">Original</div>', unsafe_allow_html=True)
                st.image(rres["original"], use_container_width=True)
            with restyled_col:
                st.markdown('<div class="section-label" style="text-align:center;">Restyled</div>', unsafe_allow_html=True)
                st.image(rres["restyled"], use_container_width=True)

                buf = io.BytesIO()
                rres["restyled"].save(buf, format="PNG")
                st.download_button(
                    label="⬇  Download Restyled Image",
                    data=buf.getvalue(),
                    file_name="adgen_restyled.png",
                    mime="image/png",
                    key="download_restyled"
                )

            if st.button("↺  Restyle Another", key="regen_restyle"):
                st.session_state.restyle_result = None
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)  # close result-card

    # ── POSTER GENERATION (Custom Input / Text Prompt modes only) ────────────
    if st.session_state.input_mode in ("custom", "text"):

        # ── Resolve active values (always defined before generate button) ────────
        if st.session_state.input_mode == "text":
            active_title    = st.session_state.get("tp_title",    "").strip()
            active_desc     = st.session_state.get("tp_desc",     "").strip()
            active_category = st.session_state.get("tp_category", "Select category")
        else:
            active_title    = st.session_state.get("product_title",    "").strip()
            active_desc     = st.session_state.get("product_desc",     "").strip()
            active_category = st.session_state.get("product_category", "Select category")

        # ── Generate Button ──────────────────────────────────────────────────────
        gen_l, gen_mid, gen_r = st.columns([1, 2, 1])
        with gen_mid:
            clicked_generate = st.button("✦  Generate Poster", key="generate")

        st.markdown(
            '<div style="text-align:center;margin-top:10px;font-size:13px;color:#77778A;">'
            'AI will generate your copy, visuals and poster layout.</div>',
            unsafe_allow_html=True
        )

        # ── Handle generate click ────────────────────────────────────────────────
        if clicked_generate:
            # Validate — active_title/desc/category already resolved above
            if not active_title:
                st.warning("Please enter a Product Title.")
                st.stop()
            if not active_desc:
                st.warning("Please add a Product Description.")
                st.stop()
            if active_category == "Select category":
                st.warning("Please select a Category.")
                st.stop()

            safety_msg = check_content_safety(active_title, active_desc)
            if safety_msg:
                st.warning(safety_msg)
                st.stop()

            rate_msg = check_rate_limit()
            if rate_msg:
                st.warning(rate_msg)
                st.stop()

            # ── Run the pipeline ─────────────────────────────────────────────
            progress_placeholder = st.empty()
            with progress_placeholder.container():
                st.markdown("""
                <div style="text-align:center;padding:32px 0 20px;">
                  <div style="font-size:28px;margin-bottom:12px;">✦</div>
                  <div style="font-size:16px;font-weight:600;color:#7167D9;margin-bottom:6px;">
                    AI is creating your poster…
                  </div>
                  <div style="font-size:13px;color:#77778A;">
                    Generating image · Writing copy · Assembling layout
                  </div>
                </div>
                """, unsafe_allow_html=True)

            try:
                result = run_pipeline(active_title, active_category, active_desc)
                st.session_state.poster_result = {
                    "poster":    result["poster"],
                    "ad_copy":   result["ad_copy"],
                    "title":     active_title,
                    "category":  active_category,
                }
                st.session_state.generation_count += 1
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Generation failed: {e}")
                st.stop()

            progress_placeholder.empty()

        # ── Show result if available ─────────────────────────────────────────────
        if st.session_state.poster_result:
            res      = st.session_state.poster_result
            poster   = res["poster"]
            ad_copy  = res["ad_copy"]

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-title">✦ Your Poster is Ready</div>'
                f'<div class="result-sub">{res["title"]} · {res["category"]}</div>',
                unsafe_allow_html=True
            )

            poster_col, copy_col = st.columns([1.2, 1], gap="large")

            with poster_col:
                st.image(poster, use_container_width=True, caption="")

                # Download button
                buf = io.BytesIO()
                poster.save(buf, format="PNG")
                st.download_button(
                    label="⬇  Download Poster",
                    data=buf.getvalue(),
                    file_name=f"adgen_{res['title'][:30].replace(' ','_')}.png",
                    mime="image/png",
                    key="download_poster"
                )

            with copy_col:
                st.markdown("""
                <div style="font-size:13px;font-weight:700;color:#7167D9;
                            text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px;">
                  ✦ AI-Generated Ad Copy
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="copy-box">
                  <div class="copy-row">
                    <div class="copy-label">Tagline</div>
                    <div class="copy-value" style="font-size:17px;font-weight:700;color:#7167D9;">
                      {ad_copy.get("tagline","—")}
                    </div>
                  </div>
                  <div style="height:1px;background:#E0DCF5;margin:14px 0;"></div>
                  <div class="copy-row">
                    <div class="copy-label">Caption</div>
                    <div class="copy-value">{ad_copy.get("caption","—")}</div>
                  </div>
                  <div style="height:1px;background:#E0DCF5;margin:14px 0;"></div>
                  <div class="copy-row">
                    <div class="copy-label">Description</div>
                    <div class="copy-value" style="font-size:13px;color:#55558A;">
                      {ad_copy.get("description","—")}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Regenerate button
                if st.button("↺  Generate Another", key="regen"):
                    st.session_state.poster_result = None
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)  # close result-card

    # ── AI Pipeline strip ─────────────────────────────────────────────────────
    st.markdown("""
    <div class="pipeline-wrap">
      <div class="pipe-step">
        <div class="pipe-icon">🔍</div>
        <div class="pipe-label">Understand<br>Product</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-icon">✍️</div>
        <div class="pipe-label">Create<br>Copy</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-icon">🎨</div>
        <div class="pipe-label">Generate<br>Visual</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-icon">📐</div>
        <div class="pipe-label">Design<br>Layout</div>
      </div>
      <div class="pipe-arrow">→</div>
      <div class="pipe-step">
        <div class="pipe-icon">🖼️</div>
        <div class="pipe-label">Final<br>Poster</div>
      </div>
    </div>
    <br><br>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ── ABOUT PAGE ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "about":

    st.markdown("""
    <div class="hero-section">
      <div class="hero-title">
        Peek inside the <span class="purple">AI</span>
      </div>
      <div class="hero-sub">
        Upload any image and see how our Autoencoder (AE) and Variational Autoencoder (VAE)
        compress and reconstruct it.
      </div>
      <div class="ai-badge">✦ Model Playground</div>
    </div>
    """, unsafe_allow_html=True)

    about_l, about_mid, about_r = st.columns([1, 6, 1])
    with about_mid:

        st.markdown("""
        <div class="main-card">
          <div class="main-card-title">Autoencoder &amp; VAE Explained</div>
          <div class="main-card-sub">
            Upload a product photo below to see both models reconstruct it side by side.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)

        # ── Reconstruction Quality (aggregate) ──────────────────────────────
        st.markdown("""
        <div class="copy-box" style="margin-top:0;">
          <div style="font-size:13px;font-weight:700;color:#7167D9;
                      text-transform:uppercase;letter-spacing:0.8px;margin-bottom:14px;">
            Reconstruction Quality (aggregate)
          </div>
          <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead>
              <tr style="border-bottom:1.5px solid #DDD8F5;">
                <th style="text-align:left;padding:8px 10px;color:#77778A;font-weight:600;">Metric</th>
                <th style="text-align:right;padding:8px 10px;color:#77778A;font-weight:600;">Autoencoder (AE)</th>
                <th style="text-align:right;padding:8px 10px;color:#77778A;font-weight:600;">VAE</th>
                <th style="text-align:right;padding:8px 10px;color:#77778A;font-weight:600;">Transformer</th>
              </tr>
            </thead>
            <tbody>
              <tr style="border-bottom:1px solid #E0DCF5;">
                <td style="padding:8px 10px;color:#303044;font-weight:500;">MSE</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.000620</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.00094</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
              </tr>
              <tr style="border-bottom:1px solid #E0DCF5;">
                <td style="padding:8px 10px;color:#303044;font-weight:500;">PSNR (dB)</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">32.982460</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">25.103167</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
              </tr>
              <tr style="border-bottom:1px solid #E0DCF5;">
                <td style="padding:8px 10px;color:#303044;font-weight:500;">SSIM</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.976403</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.9477</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
              </tr>
              <tr style="border-bottom:1px solid #E0DCF5;">
                <td style="padding:8px 10px;color:#303044;font-weight:500;">KL Divergence</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.98367</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
              </tr>
              <tr style="border-bottom:1px solid #E0DCF5;">
                <td style="padding:8px 10px;color:#303044;font-weight:500;">Mean BLEU</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.0810</td>
              </tr>
              <tr>
                <td style="padding:8px 10px;color:#303044;font-weight:500;">Mean ROUGE-L</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
                <td style="text-align:right;padding:8px 10px;color:#C0BCE0;">—</td>
                <td style="text-align:right;padding:8px 10px;color:#303044;">0.1673</td>
              </tr>
            </tbody>
          </table>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

        exp_ae, exp_vae = st.columns(2, gap="large")
        with exp_ae:
            st.markdown("""
            <div class="float-card" style="margin-bottom:0;">
              <div class="float-card-icon">🧩</div>
              <div class="float-card-title">Autoencoder (AE)</div>
              <div class="float-card-sub" style="font-size:12px;line-height:1.5;">
                Compresses an image down to a small feature vector, then reconstructs it
                back — learns the core visual structure of a product.
              </div>
            </div>
            """, unsafe_allow_html=True)
        with exp_vae:
            st.markdown("""
            <div class="float-card" style="margin-bottom:0;">
              <div class="float-card-icon">◇</div>
              <div class="float-card-title">Variational Autoencoder (VAE)</div>
              <div class="float-card-sub" style="font-size:12px;line-height:1.5;">
                Encodes an image into a smooth latent distribution rather than a single
                point — this is the same kind of encoder that powers the diffusion model's
                image generation.
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Upload an Image</div>', unsafe_allow_html=True)
        about_upload = st.file_uploader(
            "Image for AE / VAE demo",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="about_image"
        )

        if about_upload:
            src_img = Image.open(about_upload)
            ae_img = vae_img = None

            with st.spinner("Running AE and VAE reconstruction…"):
                try:
                    ae_img = run_ae_reconstruction(src_img)
                except Exception as e:
                    st.error(f"AE reconstruction failed: {e}")
                try:
                    vae_img = run_vae_reconstruction(src_img)
                except Exception as e:
                    st.error(f"VAE reconstruction failed: {e}")

            if ae_img or vae_img:
                st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
                orig_col, ae_col, vae_col = st.columns(3, gap="medium")
                with orig_col:
                    st.markdown('<div class="section-label" style="text-align:center;">Original</div>', unsafe_allow_html=True)
                    st.image(src_img, use_container_width=True)
                with ae_col:
                    st.markdown('<div class="section-label" style="text-align:center;">AE Reconstruction</div>', unsafe_allow_html=True)
                    if ae_img:
                        st.image(ae_img, use_container_width=True)
                with vae_col:
                    st.markdown('<div class="section-label" style="text-align:center;">VAE Reconstruction</div>', unsafe_allow_html=True)
                    if vae_img:
                        st.image(vae_img, use_container_width=True)
        else:
            st.markdown("""
            <div class="upload-box">
              <div class="upload-icon">↑</div>
              <div class="upload-label">Upload an image to see AE &amp; VAE in action</div>
              <div class="upload-hint">PNG, JPG or WEBP</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
