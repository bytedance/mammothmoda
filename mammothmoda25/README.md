# MammothModa2.5 Inference Usage Instructions

## 1. Environment Installation
```bash
pip install -e .
pip install "torch>=2.0" "diffusers>=0.36.0" "transformers>=4.40.0" accelerate pillow imageio-ffmpeg numpy
```

## 2. Model Weights Download
1. Download the model weight package, including the main model and VAE decoder weights.
2. After decompression, the weight directory structure is as follows:
```
weights_hf/
├── vae/
├── transformer/
├── text_encoder
├── tokenizer
├── configuration.json
└── model_index.json
```

## 3. Run Inference Script (GPU)
```bash
python infer_mammoth25.py \
  --model_dir weights_hf \
  --prompt "A close-up shot of a fox cautiously approaching the camera." \
  --output outputs/out.mp4
```

## 4. Video Editing (Video-to-Video) Example
`infer_mammoth25.py` supports providing an input `--video` (mp4) or an image (png/jpg) as the starting point, then editing it with a prompt.

If you use `--video` (mp4), install the extra dependency first:
```bash
pip install torchvision decord
```

### Edit an input video (mp4)
```bash
python infer_mammoth25.py \
  --model_dir "weights_hf" \
  --vae_dir "weights_hf/vae" \
  --prompt "Remove the dog，keep the backgroud unchanged" \
  --video ./examples/video.mp4 \
  --num_frames 81 \
  --num_inference_steps 30 \
  --guidance_scale 3.0 \
  --output ./examples/video_edited.mp4
```

### Edit an input image (png/jpg)

```bash
python infer_mammoth25.py \
  --model_dir "weights_hf" \
  --vae_dir "weights_hf/vae" \
  --prompt "Change the word \"NPIS\" to \"CVPR\" " \
  --video ./examples/image_text.png \
  --num_frames 1 \
  --num_inference_steps 30 \
  --guidance_scale 3.0 \
  --output ./examples/image_text_edited.png

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
