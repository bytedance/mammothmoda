# MammothModa2.5 Inference Usage Instructions
## 1. Environment Installation
```bash
pip install -e .
pip install "torch>=2.0" "diffusers>=0.36.0" "transformers>=4.40.0" accelerate pillow imageio-ffmpeg numpy

```

## 2. Model Weights Download
1. Download the model weight package, including the main model and VAE decoder weights.
2. After decompression, the weight directory structure is as follows (fixed path):
```
weights_hf/
├── vae/
├── transformer/
├── text_encoder
├── tokenizer
├── configuration.json
└── model_index.json
```

## 3. Run Inference Script
### GPU
```bash
python infer_mammoth25_gpu.py \
  --model_dir weights_hf \
  --prompt "A close-up shot of a fox cautiously approaching the camera." \
  --output outputs/out.mp4
```
