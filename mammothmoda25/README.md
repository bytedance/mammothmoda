# MammothModa2.5 Inference Usage Instructions
## 1. Environment Installation
```bash
pip install diffusers==0.36.0 torch accelerate pillow imageio-ffmpeg numpy --upgrade
```

## 2. Model Weights Download
1. Download the model weight package, including the main model and VAE decoder weights.
2. After decompression, the weight directory structure is as follows:
```
weights_diffusers/
├── vae/
├── transformer/
├── text_encoder
├── tokenizer
├── configuration.json
└── model_index.json
```

## 3. Run Inference Script
### Full Inference Script: `infer_mammoth_moda25.py`

### Execution Command
```bash
cd mammothmoda_25
python infer_mammoth_moda25.py
```