# AdGen AI — Advertisement Poster Generator

Turn a product description (or an uploaded product photo) into a complete, professional advertisement poster — ad copy, product visual, and layout — using a pipeline of four AI models behind a single Streamlit app.

## Features

- **Text Prompt / Custom Input modes** — describe your product in free text, or fill in structured fields (title, category, brand, status) with an optional product photo upload.
- **AE-based upload quality gate** — uploaded photos are checked against a trained autoencoder; blurry, corrupted, or off-domain images are flagged before running generation.
- **Restyle Photo (image-to-image)** — upload your exact product photo and describe a restyle ("clean white studio background", "outdoor lifestyle look"); the product stays recognizable while the style/background changes, via VAE-latent-space img2img.
- **One-click poster generation** — fine-tuned Transformer writes tagline/caption/description, SD-Turbo generates the product visual, and the result is composed into a downloadable poster.
- **About / Model Playground page** — upload any image to see side-by-side AE and VAE reconstructions, plus an aggregate model-quality metrics table (MSE, PSNR, SSIM, KL divergence, BLEU, ROUGE-L).

## Models

| Model | Role |
|---|---|
| Autoencoder (AE) | Upload quality/validation check |
| Variational Autoencoder (VAE) | Image latent-space encoding — powers image-to-image restyling |
| Fine-tuned Transformer | Ad copy generation (tagline, caption, description) |
| SD-Turbo (Stable Diffusion, distilled) | Product visual generation |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r Requirement.txt
streamlit run src/app.py
```

> **Model weights are not included in this repository** (several GB total — over GitHub's file size limits). Place trained weights under `Model/` matching the structure referenced in `src/postermaking.py` and `src/app.py`:
> - `Model/sd-turbo-saved-model/...` — SD-Turbo diffusion pipeline
> - `Model/transformer_best (2)/` — fine-tuned ad-copy Transformer
> - `Model/vae_best/` — VAE (diffusers `AutoencoderKL` format)
> - `Model/autoencoder_full_1.keras` — trained Keras autoencoder

## Project Structure

```
src/
  app.py            # Streamlit UI
  postermaking.py   # Model loading + generation pipeline
fonts/              # Poster text fonts
Requirement.txt      # Python dependencies
```

## Tech Stack

Streamlit · PyTorch · TensorFlow/Keras · HuggingFace Diffusers & Transformers
