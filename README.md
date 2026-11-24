# AdGen AI — AI-Powered Advertisement Poster Generator

## Project Team

**Project Title:** AdGen AI — AI-Powered Advertisement Poster Generator

**Team Members:**

1. **Vaibhav Kawa** — Exam Seat No. 202301040040
2. **Harshada Kolekar** — Exam Seat No. 202301040052
3. **Ashwini Arude** — Exam Seat No. 202301040218
4. **Shreya Newaskar** — Exam Seat No. 202301040240

---

## Project Description

AdGen AI is an AI-powered advertisement poster generator that transforms a product description or uploaded product photograph into a complete, professional advertisement poster.

The application integrates four AI models into a single Streamlit interface to generate advertisement copy, create product visuals, validate uploaded images, and restyle existing product photographs.

## Features

- **Text Prompt / Custom Input Modes:** Describe a product using free text or enter structured information such as title, category, brand, and product status, with an optional product photo upload.
- **Autoencoder-Based Image Quality Validation:** Uploaded photographs are checked using a trained Autoencoder (AE) to identify blurry, corrupted, or out-of-domain images before generation.
- **Image Restyling:** Upload an existing product photograph and describe the desired appearance, such as a clean white studio background or an outdoor lifestyle setting. VAE latent-space image-to-image processing helps preserve product identity while modifying its style and background.
- **One-Click Poster Generation:** A fine-tuned Transformer generates the advertisement tagline, caption, and description. SD-Turbo generates the product visual, and the application combines the generated content into a downloadable advertisement poster.
- **About / Model Playground:** Explore side-by-side Autoencoder and VAE image reconstructions and view model evaluation metrics, including MSE, PSNR, SSIM, KL Divergence, BLEU, and ROUGE-L.

## Models Used

| Model | Purpose |
|---|---|
| Autoencoder (AE) | Uploaded image quality validation |
| Variational Autoencoder (VAE) | Image encoding and image-to-image restyling |
| Fine-tuned Transformer | Advertisement tagline, caption, and description generation |
| SD-Turbo (Stable Diffusion) | Product visual generation |

## Technology Stack

- **Programming Language:** Python
- **Frontend / Interface:** Streamlit
- **Deep Learning:** PyTorch, TensorFlow, Keras
- **Generative AI:** HuggingFace Diffusers and Transformers
- **Image Generation:** Stable Diffusion Turbo
- **Image Processing:** PIL, NumPy

## Project Structure

```text
AdGen-AI/
│
├── src/
│   ├── app.py                 # Streamlit application interface
│   └── postermaking.py        # Model loading and poster generation pipeline
│
├── Model/
│   ├── sd-turbo-saved-model/
│   ├── transformer_best (2)/
│   ├── vae_best/
│   └── autoencoder_full_1.keras
│
├── fonts/                     # Fonts used in poster design
│
├── Requirement.txt            # Python dependencies
│
└── README.md                  # Project documentation
```

## Installation and Setup

Follow these steps to install the required dependencies and run the application locally.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Navigate to the project directory:

```bash
cd AdGen-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**For Windows:**

```bash
venv\Scripts\activate
```

**For Linux / macOS:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r Requirement.txt
```

### 5. Download and Configure Model Weights

The trained model weights are not included in this GitHub repository because of their large file sizes.

Place the downloaded model weights inside the `Model/` directory using the following structure:

- `Model/sd-turbo-saved-model/` — SD-Turbo diffusion pipeline.
- `Model/transformer_best (2)/` — Fine-tuned advertisement copy Transformer.
- `Model/vae_best/` — VAE model in Diffusers AutoencoderKL format.
- `Model/autoencoder_full_1.keras` — Trained Keras Autoencoder.

Ensure that all model files are placed in the paths referenced by `src/app.py` and `src/postermaking.py`.

### 6. Run the Application

Execute the following command from the project root directory:

```bash
streamlit run src/app.py
```

The application will open in your default browser. If it does not open automatically, visit:

```text
http://localhost:8501
```

## How to Use

1. Launch the AdGen AI application.
2. Select Text Prompt or Custom Input mode.
3. Enter the product details or upload a product photograph.
4. Allow the Autoencoder to validate the uploaded image.
5. Generate advertisement content and product visuals.
6. Optionally use the Restyle Photo feature to modify the product photograph.
7. Preview the generated advertisement poster.
8. Download the final advertisement poster.

## Model Evaluation

The Model Playground provides quantitative evaluation metrics for the implemented models.

| Metric | Purpose |
|---|---|
| MSE | Measures reconstruction error |
| PSNR | Measures image reconstruction quality |
| SSIM | Measures structural similarity |
| KL Divergence | Evaluates latent distribution differences |
| BLEU | Evaluates generated text quality |
| ROUGE-L | Measures textual similarity |

**Transformer Evaluation Results:**
- Mean BLEU: 0.081
- Mean ROUGE-L: 0.167

Generated advertisement content should be reviewed and edited before practical use.

## Ethical Use and Deployment Considerations

- **AI-Generated Disclosure:** Every generated poster and restyled image includes an "AI-generated with AdGen AI" label.
- **Prompt-Level Content Filtering:** User-provided product information and restyling prompts are checked using a basic keyword blocklist through `check_content_safety` in `src/app.py`.
- **Image Safety:** An image-level content moderation system is not currently integrated. Public deployment requires an appropriate content classifier to review generated images.
- **EXIF Metadata Removal:** Uploaded images are reconstructed from raw pixels using `strip_exif` to remove GPS and device metadata.
- **Generation Rate Limit:** A soft limit of 15 generations per browser session is configured through `MAX_GENERATIONS_PER_SESSION`. Production deployment should include server-side rate limiting and authentication.
- **Human Review:** Generated advertisement copy should be reviewed before publication.
- **Model Licensing:** Verify the licensing terms of `stabilityai/sd-turbo` and other model components before commercial or public deployment.

## Future Enhancements

- Improve advertisement copy generation quality.
- Integrate image-level safety and content moderation.
- Add additional poster templates and customization options.
- Improve product consistency during image restyling.
- Implement secure authentication and server-side rate limiting.
- Support multiple product categories and advertising formats.

## Conclusion

AdGen AI demonstrates how multiple generative AI and deep learning models can work together to automate advertisement creation. By integrating image validation, latent-space image restyling, Transformer-based text generation, and Stable Diffusion-based visual generation, the project provides an end-to-end platform for creating professional product advertisement posters.
