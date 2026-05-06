<div align="center">

# Mamoda2

**A Unified AR-Diffusion Model for Multimodal Understanding and Generation**

📑 [Technical Report (arXiv:2511.18262)](https://arxiv.org/abs/2511.18262)

</div>

## Overview

Mamoda2 is a unified autoregressive-diffusion (AR-Diffusion) model that integrates multimodal understanding and generation within a single architecture. Built on Qwen3-VL with dedicated generation experts and a dense DiT, it supports **text-to-image generation**, **image editing**, and **multimodal understanding** in one model.

## Showcases

### Text-to-Image & Image Editing

<div align="center">
  <img src='../doc/mammoth.png' alt="Mamoda2 Showcases" style="max-width: 80%; height: auto;">
</div>

## Models

| Model | Download | Architecture | Capabilities |
|-------|----------|--------------|--------------|
| Mamoda2-Dev | [🤗 HuggingFace](https://huggingface.co/bytedance-research/MammothModa2-Dev) | Qwen3VL-8B + 3B gen experts + 2B dense DiT | Image Generation & Editing |
| Mamoda2-Preview | [🤗 HuggingFace](https://huggingface.co/bytedance-research/MammothModa2-Preview) | Qwen25VL-7B + 3B gen experts + 2B dense DiT | Image Generation (use `qwen25vl` branch) |

## Benchmark Results

### Text-to-Image

| Model | Model Size | GenEval | DPGBench |
|-------|------------|---------|----------|
| **Generation** |
| SDXL | - | 0.55 | 74.65 |
| DALL-E 3 | - | 0.67 | 83.50 |
| FLUX.1-dev | - | 0.67 | 84.00 |
| SD3.5-Medium* | - | 0.65 | 83.86 |
| **Unified** |
| Emu3 | 8B | 0.66 | 80.60 |
| Janus-Pro | 7B | 0.80 | 84.19 |
| MetaQuery-XL | 7B + 1.6B | 0.80 | 82.05 |
| UniWorld-V1 | 7B + 12B | 0.84 | 81.38 |
| Blip3-o-8B | 7B + 1.4B | 0.84 | 81.60 |
| OmniGen2 | 3B + 4B | 0.86 | 83.57 |
| Ovis-U1 | 2.4B + 1.2B | 0.89 | 83.72 |
| UniPic2 | 7B + 2B | 0.90 | 83.79 |
| BAGEL | 7B + 7B | 0.88 | 85.07 |
| Show-o2 | 7B | 0.76 | 86.14 |
| GPT-4o | - | 0.84 | 86.23 |
| **Mamoda2** | **8B + (3B + 2B)** | **0.87** | **87.2** |

> Model sizes in "A + B" format indicate separate understanding (A) and generation (B) parameters. Mamoda2 uses 8B for understanding, 3B AR parameters in the MLLM backbone, and 2B in the DiT component for generation.

## Usage

### Installation

```bash
git clone https://github.com/bytedance/mammothmoda.git
cd mammothmoda

# Install dependencies
uv sync --frozen
```

### Text-to-Image Generation

```python
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from mammothmoda2.model import DEFAULT_NEGATIVE_PROMPT, Mammothmoda2Model
from mammothmoda2.utils import decode_diffusion_image

# Load model and processor
model = Mammothmoda2Model.from_pretrained(
    "bytedance-research/MammothModa2-Dev",
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
    t2i_generate=True,
).to("cuda")
processor = AutoProcessor.from_pretrained(
    "bytedance-research/MammothModa2-Dev",
    t2i_generate=True,
    ar_height=32,
    ar_width=32,
)

# Prepare inputs
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "A modern city skyline at sunset with sailboats on the water."}],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    num_images_per_prompt=4,
    cfg_scale=7.0,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    padding=True,
    padding_side="left",
    return_tensors="pt",
    return_token_type_ids=False,
).to("cuda")

# Generate
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    generated_ids, attention_mask = model.generate(**inputs)
    decode_diffusion_image(
        input_ids=inputs.input_ids,
        generated_ids=generated_ids,
        attention_mask=attention_mask,
        negative_ids=inputs.get("negative_ids", None),
        negative_mask=inputs.get("negative_mask", None),
        model=model,
        tokenizer=processor.tokenizer,
        output_dir="./mamoda2_t2i_output",
        num_images_per_prompt=4,
        text_guidance_scale=9.0,
        vae_scale_factor=16,
        cfg_range=(0.0, 1.0),
        num_inference_steps=50,
        height=1024,
        width=1024,
    )
```

### Multimodal Understanding

```python
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from mammothmoda2.model import Mammothmoda2Model

model = Mammothmoda2Model.from_pretrained(
    "bytedance-research/MammothModa2-Dev",
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
).to("cuda")
processor = AutoProcessor.from_pretrained("bytedance-research/MammothModa2-Dev")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "path/to/image.png"},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    padding_side="left",
    return_tensors="pt",
    return_token_type_ids=False,
).to("cuda")

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    generated_ids = model.generate(**inputs)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_texts = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
print(output_texts)
```

## Acknowledgement

- [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

## Citation

```bibtex
@article{shen2025mammothmoda2,
    title={MammothModa2: A Unified AR-Diffusion Framework for Multimodal Understanding and Generation},
    author={Shen, Tao and Wan, Xin and Chen, Taicai and Zhang, Rui and Pan, Junwen and Lu, Dawei and Lei, Fanding and Lu, Zhilin and Yang, Yunfei and Cheng, Chen and She, Qi and Liu, Chang and Sun, Zhenbang},
    journal={arXiv preprint arXiv:2511.18262},
    year={2025},
    url={https://arxiv.org/abs/2511.18262}
}
```
