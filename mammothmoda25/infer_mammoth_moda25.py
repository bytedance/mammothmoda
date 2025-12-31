import torch
import sys
import importlib

if 'diffusers.models.transformers.transformer_wan' in sys.modules:
    importlib.reload(sys.modules['diffusers.models.transformers.transformer_wan'])
import diffusers.models.transformers.transformer_wan as transformer_wan_module

from transformer_mammothmoda import WanTransformer3DModel as MammothModa25Transformer3DModel
from transformer_mammothmoda import WanTransformerBlock as MammothModa25TransformerBlock
from transformer_mammothmoda import WanTimeTextImageEmbedding as MammothModa25TimeTextImageEmbedding

transformer_wan_module.WanTransformer3DModel = MammothModa25Transformer3DModel
transformer_wan_module.WanTransformerBlock = MammothModa25TransformerBlock
transformer_wan_module.WanTimeTextImageEmbedding = MammothModa25TimeTextImageEmbedding

from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from pipeline_mammothmoda import MammothModa25Pipeline

vae = AutoencoderKLWan.from_pretrained(
    "/mnt/bn/genai-video1/zsx/code/crane_infer_diffusers/weights_hf/vae",
    subfolder="vae",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

pipe = MammothModa25Pipeline.from_pretrained(
    "/mnt/bn/genai-video1/zsx/code/crane_infer_diffusers/weights_hf",
    vae=vae,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)
pipe.to('cuda')

prompt = "A close-up shot of a fox cautiously approaching the camera, sniffing at the lens curiously. The camera captures the fox’s inquisitive expression, the twitching of its nose, and the rustling of leaves under its paws."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

torch.manual_seed(1234)

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=848,
    num_frames=93,
    guidance_scale=4.5
).frames[0]

export_to_video(output, "Output_MammothModa25.mp4", fps=24)