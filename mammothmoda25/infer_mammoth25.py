from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Optional

import torch


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_dtype(value: str) -> torch.dtype:
    v = value.strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16", "half"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _patch_wan_transformer() -> None:
    import diffusers.models.transformers.transformer_wan as transformer_wan_module

    from src import transformer_mammothmoda as mammothmoda_module

    replacements = {
        "WanTransformer3DModel": mammothmoda_module.WanTransformer3DModel,
        "WanTransformerBlock": mammothmoda_module.WanTransformerBlock,
        "WanTimeTextImageEmbedding": mammothmoda_module.WanTimeTextImageEmbedding,
    }
    for attr_name, replacement in replacements.items():
        setattr(transformer_wan_module, attr_name, replacement)


def _build_pipeline(
    model_dir: str,
    vae_dir: str,
    vae_subfolder: str,
    pipe_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    device: str,
    low_cpu_mem_usage: bool,
) -> "MammothModa25Pipeline":
    _patch_wan_transformer()

    from diffusers import AutoencoderKLWan

    from src import pipeline_mammothmoda as pm
    MammothModa25Pipeline = pm.MammothModa25Pipeline

    vae = AutoencoderKLWan.from_pretrained(
        vae_dir,
        subfolder=vae_subfolder,
        torch_dtype=vae_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    pipe = MammothModa25Pipeline.from_pretrained(
        model_dir,
        vae=vae,
        torch_dtype=pipe_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    pipe.to(device)
    return pipe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--vae_dir", type=str, default=None)
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pipe_dtype", type=str, default="bf16")
    parser.add_argument("--vae_dtype", type=str, default="fp32")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but device starts with 'cuda'.")

    model_dir = os.path.expanduser(args.model_dir)
    vae_dir = os.path.expanduser(args.vae_dir) if args.vae_dir is not None else model_dir
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)
    pipe = _build_pipeline(
        model_dir=model_dir,
        vae_dir=vae_dir,
        vae_subfolder=args.vae_subfolder,
        pipe_dtype=_parse_dtype(args.pipe_dtype),
        vae_dtype=_parse_dtype(args.vae_dtype),
        device=args.device,
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
    )

    output_frames = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
    ).frames[0]

    from diffusers.utils import export_to_video

    export_to_video(
        output_frames,
        str(output_path),
        quality=int(args.quality),
        fps=int(args.fps),
    )


if __name__ == "__main__":
    main()
