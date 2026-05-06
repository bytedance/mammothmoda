<div align="center">

# Mamoda2.5

**Unified Visual Generation & Editing with Fine-Grained MoE DiT**

📑 [Technical Report (arXiv:2605.02641)](https://arxiv.org/abs/2605.02641)

</div>

## Overview

Mamoda2.5 is the latest evolution of the Mamoda family, powered by Qwen3-VL and a **fine-grained Mixture-of-Experts (MoE) Diffusion Transformer (DiT)** with 128 routed experts (25B total, ~3B active). One unified model supports **text-to-video, video editing, text-to-image, and image editing** with state-of-the-art performance and compute-efficient inference.

**Key highlights:**
- **25B-A3B MoE DiT** with 128 experts and Top-8 routing — only ~12% parameters active per forward pass
- **12x faster inference** than Wan2.2 A14B on a single device
- **#1 on OpenVE-Bench, #1 on FiVE-Bench, best overall on Reco-Bench** for video editing

## Model

| Model | Download | Architecture | Capabilities |
|-------|----------|--------------|--------------|
| Mamoda2.5-25B-A3B | Coming Soon | Qwen3-VL + 25B-A3B MoE DiT (E128A8) | Video Generation, Video Editing, Image Editing |

## Benchmark Results

### Text-to-Video (VBench 2.0)

| Model | Total |
|-------|-------|
| **Proprietary** |
| Sora-480p | 58.38 |
| Kling1.6 | 59.00 |
| Vidu Q1 | 62.70 |
| Seedance 1.0 Pro | 59.81 |
| Veo3 | **66.72** |
| **Open Source** |
| HunyuanVideo | 55.30 |
| Wan2.1 | 60.20 |
| LongCat-Video | 62.11 |
| **Mamoda2.5** | **61.64** |

### Video Editing

| Model | OpenVE-Bench | FiVE-Acc |
|-------|-------------|----------|
| **Proprietary** |
| Kling O1 | 3.69 | - |
| **Open Source** |
| VACE-14B | 1.65 | - |
| Wan-Edit | - | 46.97 |
| Omni-Video2 | - | 73.53 |
| VInO | 3.21 | - |
| **Mamoda2.5** | **3.86** | **87.41** |

### Image Editing

| Model | ImgEdit Avg. | GEdit-EN Overall |
|-------|-------------|-----------------|
| **Proprietary** |
| Gemini 2.5 | 4.30 | 7.17 |
| GPT-4o | 4.30 | 7.48 |
| Seedream 4 | 4.46 | 7.72 |
| **Open Source** |
| Flux-Kontext-Dev | 4.09 | 6.53 |
| Step1x-Edit | 4.01 | 6.87 |
| Mamoda2 | 4.06 | 6.82 |
| VInO | 4.18 | 6.88 |
| **Mamoda2.5** | **4.22** | **7.05** |

## Usage

### Installation

```bash
pip install -e .
pip install "torch>=2.0" "diffusers>=0.36.0" "transformers>=4.40.0" accelerate pillow imageio-ffmpeg numpy
```

### Text-to-Video Generation

```bash
python infer_mammoth25.py \
  --model_dir weights_hf \
  --prompt "A close-up shot of a fox cautiously approaching the camera." \
  --output outputs/out.mp4
```

### Video Editing

```bash
# Install extra dependency for video input
pip install torchvision decord

python infer_mammoth25.py \
  --model_dir "weights_hf" \
  --vae_dir "weights_hf/vae" \
  --prompt "Remove the dog, keep the background unchanged" \
  --video ./examples/video.mp4 \
  --num_frames 81 \
  --num_inference_steps 30 \
  --guidance_scale 3.0 \
  --output ./examples/video_edited.mp4
```

### Image Editing

```bash
python infer_mammoth25.py \
  --model_dir "weights_hf" \
  --vae_dir "weights_hf/vae" \
  --prompt "Remove the dog" \
  --video ./examples/image.png \
  --num_frames 1 \
  --num_inference_steps 30 \
  --guidance_scale 3.0 \
  --output ./examples/image_edited.png
```

### Model Weights

Download the model weight package (main model + VAE decoder). Expected directory structure:

```
weights_hf/
├── vae/
├── transformer/
├── text_encoder/
├── tokenizer/
├── configuration.json
└── model_index.json
```

## Acknowledgement

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)

## Citation

```bibtex
@article{mamoda25,
    title={Mamoda2.5: Unified Visual Generation and Editing with Fine-Grained MoE DiT},
    journal={arXiv preprint arXiv:2605.02641},
    year={2025},
    url={https://arxiv.org/abs/2605.02641}
}
```
