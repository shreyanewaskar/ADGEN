# -*- coding: utf-8 -*-
"""
AdGen AI — postermaking pipeline
SD-Turbo (local, float32) + fine-tuned Transformer → advertisement poster

NOTE — GTX 16-series (1650/1660) fp16 NaN bug:
  These cards lack the tensor-core fp16 path that Turing/Ampere cards have.
  Loading the pipeline in float32 is the correct workaround — the UNet,
  VAE, and text encoder all run in float32, avoiding NaN latents entirely.
  Attention slicing keeps VRAM within 4 GB.
"""

import os
import re
import urllib.request
import urllib.parse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, AutoPipelineForImage2Image

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so they work from any cwd
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)   # project root

DIFFUSION_MODEL_ID = os.path.join(
    _ROOT, "Model", "sd-turbo-saved-model",
    "content", "drive", "MyDrive", "Adgen_dataset",
    "diffusion_poster_results", "sd-turbo-saved-model"
)
TRANSFORMER_DIR = os.path.join(_ROOT, "Model", "transformer_best (2)")
FONT_DIR        = os.path.join(_ROOT, "fonts")
METADATA_CSV    = os.path.join(_ROOT, "metadata_clean.csv")

CATEGORY_CHOICES_FALLBACK = [
    "Home & Garden", "Sporting Goods", "Electronics",
    "Health & Beauty", "Toys & Games"
]

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[AdGen] Device: {device}")
if device == "cuda":
    print(f"[AdGen] GPU: {torch.cuda.get_device_name(0)}")
else:
    print("[AdGen] WARNING: no GPU — diffusion will be slow.")

# ---------------------------------------------------------------------------
# 1. Load SD-Turbo (float32 — GTX 16-series fp16 NaN workaround)
# ---------------------------------------------------------------------------
print("[AdGen] Loading SD-Turbo from local model …")

# Load entirely on CPU first — avoids VRAM overflow during component loading
pipe = StableDiffusionPipeline.from_pretrained(
    DIFFUSION_MODEL_ID,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True,
)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

if device == "cuda":
    # Keep text_encoder on CPU (saves ~500 MB VRAM) — it only runs once per
    # generation and is fast enough on CPU.
    # Move UNet + VAE to GPU for fast diffusion steps.
    pipe.text_encoder = pipe.text_encoder.to("cpu")
    pipe.unet         = pipe.unet.to(device)
    pipe.vae          = pipe.vae.to(device)
    pipe.enable_attention_slicing(1)
    print("[AdGen] SD-Turbo ready — UNet+VAE on GPU, text_encoder on CPU (float32).")
else:
    pipe = pipe.to(device)
    print("[AdGen] SD-Turbo ready (CPU, float32).")

# ---------------------------------------------------------------------------
# 2. Load fine-tuned Transformer for ad copy
# ---------------------------------------------------------------------------
TRANSFORMER_AVAILABLE = os.path.exists(TRANSFORMER_DIR)
if TRANSFORMER_AVAILABLE:
    print(f"[AdGen] Loading Transformer from {TRANSFORMER_DIR} …")
    text_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    text_model = AutoModelForCausalLM.from_pretrained(
        TRANSFORMER_DIR,
        torch_dtype=torch.float32,
    ).to(device)
    text_model.eval()
    print("[AdGen] Transformer ready.")
else:
    print(f"[AdGen] Transformer not found — using rule-based fallback.")

try:
    import pandas as pd
    df = pd.read_csv(METADATA_CSV)
    CATEGORY_CHOICES = sorted(df["category"].unique().tolist())
except Exception:
    CATEGORY_CHOICES = CATEGORY_CHOICES_FALLBACK

# ---------------------------------------------------------------------------
# 3. Image generation
# ---------------------------------------------------------------------------
AD_STYLE_SUFFIX = (
    ", professional product photography, studio lighting, "
    "advertisement, high quality, sharp focus, commercial photo"
)
NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, deformed, "
    "extra limbs, cropped, out of frame"
)

def build_image_prompt(title, category, description=""):
    base = f"{title}, {category.lower()}"
    if description:
        base += f", {description[:120]}"
    return base + AD_STYLE_SUFFIX

def generate_ad_image(title, category, description="", seed=None,
                      num_inference_steps=4, guidance_scale=1.0):
    prompt = build_image_prompt(title, category, description)
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    with torch.no_grad():
        # Encode text on CPU (text_encoder lives there to save VRAM)
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(
            text_inputs.input_ids.to("cpu")
        )[0].to(device)   # move embeddings to GPU for UNet

        # Unconditional embeddings for guidance
        uncond_inputs = pipe.tokenizer(
            NEGATIVE_PROMPT,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg_embeds = pipe.text_encoder(
            uncond_inputs.input_ids.to("cpu")
        )[0].to(device)

        # Run full pipeline using pre-computed embeddings
        latents = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
        ).images   # float32, shape (1, 4, 64, 64)

        latents = latents / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        pixel_array = (
            decoded[0].permute(1, 2, 0).cpu().numpy() * 255
        ).round().astype("uint8")

    return Image.fromarray(pixel_array), prompt

# ---------------------------------------------------------------------------
# 3b. Image-to-image restyle (VAE-conditioned) — keeps an uploaded product
#     recognizable while a prompt steers background/style changes. Reuses
#     the already-loaded SD-Turbo UNet/VAE/text-encoder — no extra model
#     load, no extra VRAM.
# ---------------------------------------------------------------------------
RESTYLE_STYLE_SUFFIX = ", professional product photography, high quality, sharp focus, commercial photo"

_img2img_pipe = None

def _get_img2img_pipe():
    global _img2img_pipe
    if _img2img_pipe is None:
        _img2img_pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    return _img2img_pipe

def restyle_image(image, prompt, strength=0.35, num_inference_steps=12,
                   guidance_scale=1.0, seed=None):
    """Image-to-image restyle: starts denoising from the uploaded image's own
    latent (not random noise), so the product stays recognizable while the
    prompt steers style/background. `strength` (0-1) controls how much the
    output is allowed to drift from the original — low = subtle, high = more
    creative freedom (and more risk of altering the product itself)."""
    img2img = _get_img2img_pipe()
    init_image = image.convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    full_prompt = prompt.strip() + RESTYLE_STYLE_SUFFIX

    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            full_prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = pipe.text_encoder(text_inputs.input_ids.to("cpu"))[0].to(device)

        uncond_inputs = pipe.tokenizer(
            NEGATIVE_PROMPT,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg_embeds = pipe.text_encoder(uncond_inputs.input_ids.to("cpu"))[0].to(device)

        latents = img2img(
            image=init_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="latent",
        ).images

        latents = latents / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        pixel_array = (
            decoded[0].permute(1, 2, 0).cpu().numpy() * 255
        ).round().astype("uint8")

    restyled = add_ai_disclosure_label(Image.fromarray(pixel_array))
    return restyled, full_prompt

# ---------------------------------------------------------------------------
# 4. Ad copy generation
# ---------------------------------------------------------------------------
def ensure_punct(s):
    s = s.strip()
    if s and s[-1] not in ".!?":
        s += "."
    return s

def generate_ad_copy(title, category, description=""):
    if not TRANSFORMER_AVAILABLE:
        return {
            "description": ensure_punct(f"The {title} is a {category.lower()} product."),
            "caption":     ensure_punct(f"Discover the {title.lower()}."),
            "tagline":     "Quality you can trust.",
        }

    prompt = f"Title: {title}\nCategory: {category}\nDescription: {description[:350]}\nGenerate:"
    inputs = text_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=180
    ).to(device)

    with torch.no_grad():
        out = text_model.generate(
            **inputs, max_new_tokens=90, num_beams=4,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
            pad_token_id=text_tokenizer.eos_token_id,
        )
    generated = text_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )

    desc_m = re.search(r"Description:\s*(.+?)(?:\n|Caption:|$)",  generated, re.S)
    cap_m  = re.search(r"Caption:\s*(.+?)(?:\n|Tagline:|$)",      generated, re.S)
    tag_m  = re.search(r"Tagline:\s*(.+?)(?:\n|$)",               generated, re.S)

    return {
        "description": ensure_punct(desc_m.group(1).strip()) if desc_m
                       else ensure_punct(f"The {title} is a {category.lower()} product."),
        "caption":     ensure_punct(cap_m.group(1).strip()) if cap_m
                       else ensure_punct(f"Discover the {title.lower()}."),
        "tagline":     tag_m.group(1).strip() if tag_m else "Quality you can trust.",
    }

# ---------------------------------------------------------------------------
# 5. Font auto-picker
# ---------------------------------------------------------------------------
os.makedirs(FONT_DIR, exist_ok=True)

FONT_LIBRARY = {
    "handwritten_playful": ("apache/permanentmarker/PermanentMarker-Regular.ttf", "PermanentMarker-Regular.ttf", None),
    "bold_impact":         ("ofl/anton/Anton-Regular.ttf",                         "Anton-Regular.ttf",           None),
    "clean_modern":        ("ofl/montserrat/Montserrat[wght].ttf",                 "Montserrat.ttf",              "Bold"),
    "elegant_luxury":      ("ofl/playfairdisplay/PlayfairDisplay[wght].ttf",       "PlayfairDisplay.ttf",         "Bold"),
    "script_romantic":     ("ofl/dancingscript/DancingScript[wght].ttf",           "DancingScript.ttf",           "Bold"),
    "friendly_rounded":    ("ofl/fredoka/Fredoka[wdth,wght].ttf",                  "Fredoka.ttf",                 "SemiBold"),
    "retro_fun":           ("ofl/bangers/Bangers-Regular.ttf",                     "Bangers-Regular.ttf",         None),
    "industrial_bold":     ("ofl/oswald/Oswald[wght].ttf",                         "Oswald.ttf",                  "Bold"),
    "casual_handwritten2": ("ofl/caveat/Caveat[wght].ttf",                         "Caveat.ttf",                  "Bold"),
}

KEYWORD_RULES = [
    (["jewelry","watch","perfume","luxury","premium","gold","diamond"],                            ["elegant_luxury"]),
    (["beauty","cosmetic","skincare","makeup","fashion","dress","apparel","jewel"],                 ["script_romantic","elegant_luxury"]),
    (["electronics","tech","gadget","smart","digital","camera","laptop","phone","headphone"],       ["bold_impact","clean_modern"]),
    (["sport","fitness","shoe","running","gym","outdoor","athletic"],                               ["industrial_bold","bold_impact"]),
    (["food","snack","coffee","cafe","restaurant","kitchen","drink","bakery"],                      ["friendly_rounded","casual_handwritten2"]),
    (["kid","toy","baby","game","fun","candy"],                                                     ["retro_fun","friendly_rounded"]),
    (["home","furniture","decor","handmade","organic","craft","garden"],                            ["casual_handwritten2","friendly_rounded"]),
]

MOOD_PREFERENCE = {
    "dark_dramatic": ["bold_impact","industrial_bold","handwritten_playful"],
    "bright_minimal": ["clean_modern","elegant_luxury"],
    "warm_vibrant":   ["friendly_rounded","casual_handwritten2","retro_fun","script_romantic"],
    "cool_techy":     ["clean_modern","bold_impact","industrial_bold"],
    "soft_neutral":   ["handwritten_playful","script_romantic"],
}

def add_ai_disclosure_label(image, text="AI-generated with AdGen AI"):
    """Stamp a small, unobtrusive 'AI-generated' disclosure label onto the
    output image. Applied to every poster/restyle result so the disclosure
    survives download/sharing, not just shown in the UI."""
    img = image.convert("RGBA")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font_size = max(11, int(w * 0.022))
    try:
        font = ImageFont.truetype(os.path.join(FONT_DIR, "Oswald.ttf"), font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(font_size * 0.55)
    box_w, box_h = text_w + pad * 2, text_h + pad * 2
    x, y = w - box_w - 14, h - box_h - 14

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=8, fill=(0, 0, 0, 140))
    odraw.text((x + pad, y + pad - bbox[1]), text, font=font, fill=(255, 255, 255, 235))

    return Image.alpha_composite(img, overlay).convert("RGB")


def ensure_font(style_key):
    repo_path, filename, instance = FONT_LIBRARY[style_key]
    local_path = os.path.join(FONT_DIR, filename)
    if not os.path.exists(local_path):
        url = "https://raw.githubusercontent.com/google/fonts/main/" + urllib.parse.quote(repo_path)
        urllib.request.urlretrieve(url, local_path)
    return local_path, instance

def analyze_image_mood(image, sample_size=150):
    small = image.convert("RGB").resize((sample_size, sample_size))
    arr   = np.asarray(small, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    brightness  = arr.mean() / 255.0
    maxc        = arr.max(axis=-1)
    minc        = arr.min(axis=-1)
    saturation  = ((maxc - minc) / (maxc + 1e-6)).mean()
    contrast    = arr.mean(axis=-1).std() / 255.0
    warmth      = (r - b).mean() / 255.0
    if   brightness < 0.35 and contrast > 0.18: mood = "dark_dramatic"
    elif saturation < 0.15 and brightness > 0.6: mood = "bright_minimal"
    elif warmth > 0.05 and saturation > 0.2:     mood = "warm_vibrant"
    elif warmth < -0.02:                          mood = "cool_techy"
    else:                                         mood = "soft_neutral"
    return {"mood": mood}

def choose_font_style(category="", title="", description="", image=None):
    blob = f"{category} {title} {description}".lower()
    candidates = None
    for keywords, styles in KEYWORD_RULES:
        if any(k in blob for k in keywords):
            candidates = styles
            break
    if candidates is None:
        candidates = list(FONT_LIBRARY.keys())
    if image is not None:
        preferred = MOOD_PREFERENCE.get(analyze_image_mood(image)["mood"], [])
        ranked    = [s for s in preferred if s in candidates] + \
                    [s for s in candidates if s not in preferred]
        style_key = ranked[0]
    else:
        style_key = candidates[0]
    local_path, instance = ensure_font(style_key)
    return style_key, local_path, instance

# ---------------------------------------------------------------------------
# 6. Poster assembly
# ---------------------------------------------------------------------------
def add_top_gradient(img, gradient_height_ratio=0.42, max_opacity=205):
    img    = img.convert("RGBA")
    w, h   = img.size
    grad_h = int(h * gradient_height_ratio)
    gradient = Image.new("L", (1, grad_h), color=0)
    for y in range(grad_h):
        alpha = int(max_opacity * (1 - y / grad_h) ** 1.6)
        gradient.putpixel((0, y), alpha)
    gradient   = gradient.resize((w, grad_h))
    alpha_mask = Image.new("L", (w, h), 0)
    alpha_mask.paste(gradient, (0, 0))
    black_layer = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    black_layer.putalpha(alpha_mask)
    return Image.alpha_composite(img, black_layer)

def wrap_text(draw, text, font, max_width, stroke_width=0):
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = (current + " " + word).strip()
        bbox  = draw.textbbox((0, 0), trial, font=font, stroke_width=stroke_width)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

def draw_emerging_text(draw, x, y, text, font, fill,
                        stroke_width, stroke_fill, shadow_offset, shadow_fill):
    draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=shadow_fill)
    draw.text((x, y), text, font=font, fill=fill,
              stroke_width=stroke_width, stroke_fill=stroke_fill)

def find_product_top(image, min_zone_frac=0.22, max_zone_frac=0.60, edge_margin_frac=0.03):
    gray    = image.convert("L")
    edges   = gray.filter(ImageFilter.FIND_EDGES)
    arr     = np.asarray(edges, dtype=np.float32)
    h, w    = arr.shape
    row_energy = arr.mean(axis=1)
    k       = max(3, h // 60)
    kernel  = np.ones(k) / k
    smoothed   = np.convolve(row_energy, kernel, mode="same")
    baseline   = np.median(smoothed[: int(h * 0.15)])
    threshold  = baseline + (smoothed.max() - baseline) * 0.28
    product_top_row = h
    for y in range(int(h * 0.10), h):
        if smoothed[y] > threshold:
            product_top_row = y
            break
    zone_frac = (product_top_row / h) - edge_margin_frac
    return max(min_zone_frac, min(max_zone_frac, zone_frac))

def make_poster(image, tagline, caption="", category="", title="", description="",
                tagline_color="white", caption_color=(235, 235, 235)):
    w, h = image.size
    style_key, font_path, variable_instance = choose_font_style(
        category, title, description, image=image
    )
    gradient_height_ratio = find_product_top(image)
    img  = add_top_gradient(image, gradient_height_ratio=gradient_height_ratio)
    draw = ImageDraw.Draw(img)
    tagline_display = tagline.rstrip(".")

    def _load(size):
        f = ImageFont.truetype(font_path, size)
        if variable_instance is not None:
            try:
                names = [n.decode() if isinstance(n, bytes) else n
                         for n in f.get_variation_names()]
                if variable_instance in names:
                    f.set_variation_by_name(variable_instance)
            except Exception:
                pass
        return f

    zone_px_height   = int(h * gradient_height_ratio)
    top_margin       = int(h * 0.04)
    available_height = max(zone_px_height - top_margin - int(h * 0.02), int(h * 0.08))

    size = max(56, int(w * 0.15))
    tagline_font, stroke_w, tagline_lines = None, 2, []
    while size >= 28:
        f   = _load(size)
        sw  = max(2, size // 22)
        lines = wrap_text(draw, tagline_display, f,
                          max_width=int(w * 0.88), stroke_width=sw)
        total_height = sum(
            (draw.textbbox((0,0), l, font=f, stroke_width=sw)[3] -
             draw.textbbox((0,0), l, font=f, stroke_width=sw)[1]) + int(size * 0.22)
            for l in lines
        )
        if total_height <= available_height:
            tagline_font, stroke_w, tagline_lines = f, sw, lines
            break
        size -= 4

    if tagline_font is None:
        size         = 28
        tagline_font = _load(size)
        stroke_w     = max(2, size // 22)
        tagline_lines = wrap_text(draw, tagline_display, tagline_font,
                                  max_width=int(w * 0.88), stroke_width=stroke_w)

    shadow_off = max(3, size // 16)
    y = top_margin
    for line in tagline_lines:
        bbox   = draw.textbbox((0, 0), line, font=tagline_font, stroke_width=stroke_w)
        text_w = bbox[2] - bbox[0]
        x      = (w - text_w) / 2
        draw_emerging_text(draw, x, y, line, tagline_font,
                           tagline_color, stroke_w, (0, 0, 0, 255),
                           shadow_off, (0, 0, 0, 160))
        y += (bbox[3] - bbox[1]) + int(size * 0.22)

    if caption:
        caption_font_size = max(18, int(w * 0.035))
        try:
            caption_font = ImageFont.truetype("DejaVuSans.ttf", caption_font_size)
        except OSError:
            caption_font = ImageFont.load_default()
        y += int(size * 0.15)
        for line in wrap_text(draw, caption, caption_font,
                              max_width=int(w * 0.8), stroke_width=1):
            bbox = draw.textbbox((0, 0), line, font=caption_font, stroke_width=1)
            x    = (w - (bbox[2] - bbox[0])) / 2
            draw.text((x, y), line, font=caption_font,
                      fill=caption_color, stroke_width=1, stroke_fill=(0, 0, 0, 230))
            y += (bbox[3] - bbox[1]) + int(caption_font_size * 0.35)

    return img.convert("RGB")

# ---------------------------------------------------------------------------
# 7. Full pipeline: text → poster
# ---------------------------------------------------------------------------
def text_to_poster(title, category, description="", seed=None):
    image, prompt = generate_ad_image(title, category, description, seed=seed)
    ad_copy       = generate_ad_copy(title, category, description)
    poster        = make_poster(image, tagline=ad_copy["tagline"],
                                category=category, title=title, description=description)
    poster        = add_ai_disclosure_label(poster)
    return {"image": image, "poster": poster, "prompt_used": prompt, "ad_copy": ad_copy}

# ---------------------------------------------------------------------------
# 8. Gradio UI (standalone mode only)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gradio as gr

    def generate_poster_ui(title, category, description):
        if not title.strip():
            return None, "Please enter a product title."
        result   = text_to_poster(title, category, description)
        ad_copy  = result["ad_copy"]
        copy_text = (
            f"Description: {ad_copy['description']}\n"
            f"Caption:     {ad_copy['caption']}\n"
            f"Tagline:     {ad_copy['tagline']}"
        )
        return result["poster"], copy_text

    demo = gr.Interface(
        fn=generate_poster_ui,
        inputs=[
            gr.Textbox(label="Product Title"),
            gr.Dropdown(choices=CATEGORY_CHOICES, label="Category"),
            gr.Textbox(label="Description", lines=3),
        ],
        outputs=[
            gr.Image(label="Generated Poster", type="pil"),
            gr.Textbox(label="Ad Copy", lines=4),
        ],
        title="AI Ad Poster Generator",
    )
    demo.launch()
