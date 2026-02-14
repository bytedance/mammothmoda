from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torchvision.transforms as transforms
from PIL import Image

if TYPE_CHECKING:
    from src.pipeline_mammothmoda import MammothModa25Pipeline


def open_video(
    file_path: str,
    start_frame_idx: int,
    num_frames: int,
    frame_interval: int = 1,
    target_height: int | None = None,
    target_width: int | None = None,
) -> torch.Tensor:
    def _resize_hw_keep_aspect_ratio(
        orig_h: int, orig_w: int, size_h: int, size_w: int
    ) -> tuple[int, int]:
        max_area = int(size_h) * int(size_w)
        if max_area <= 0:
            return orig_h, orig_w
        aspect_ratio = orig_h / max(orig_w, 1)
        new_h = int(round(math.sqrt(max_area * aspect_ratio)))
        new_w = int(round(math.sqrt(max_area / max(aspect_ratio, 1e-12))))
        return max(new_h, 1), max(new_w, 1)

    if file_path.endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(file_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img) * 255
        img_tensor = img_tensor.byte()
        _, h, w = img_tensor.shape

        if (
            target_height is not None
            and target_width is not None
            and target_height > 0
            and target_width > 0
        ):
            new_h, new_w = _resize_hw_keep_aspect_ratio(
                h, w, target_height, target_width
            )
        else:
            new_h, new_w = (h // 16) * 16, (w // 16) * 16

        if new_h != h or new_w != w:
            resized = torch.nn.functional.interpolate(
                img_tensor.float().unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )
            img_tensor = resized.squeeze(0).round().clamp(0, 255).to(torch.uint8)
        video_data = img_tensor.unsqueeze(0)
        return video_data

    import decord

    decord_vr = decord.VideoReader(file_path, ctx=decord.cpu(0), num_threads=1)

    total_frames = len(decord_vr)
    frame_indices = list(
        range(
            start_frame_idx,
            min(start_frame_idx + num_frames * frame_interval, total_frames),
            frame_interval,
        )
    )

    if len(frame_indices) == 0:
        raise ValueError(
            "No frames selected. Check your start_frame_idx and num_frames."
        )

    if len(frame_indices) < num_frames:
        frame_indices = frame_indices[: (len(frame_indices) - 1) // 4 * 4 + 1]

    if len(frame_indices) > 1000:
        raise ValueError("Frames has to be less than or equal to 1000")

    video_data = decord_vr.get_batch(frame_indices).asnumpy()
    video_data = torch.from_numpy(video_data)
    video_data = video_data.permute(0, 3, 1, 2)

    if (
        target_height is not None
        and target_width is not None
        and target_height > 0
        and target_width > 0
    ):
        t, c, h, w = video_data.shape
        new_h, new_w = _resize_hw_keep_aspect_ratio(h, w, target_height, target_width)
        if new_h != h or new_w != w:
            video_data = (
                torch.nn.functional.interpolate(
                    video_data.float(),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
    else:
        t, c, h, w = video_data.shape
        new_h, new_w = (h // 16) * 16, (w // 16) * 16
        if new_h != h or new_w != w:
            video_data = (
                torch.nn.functional.interpolate(
                    video_data.float(),
                    size=(new_h, new_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )

    return video_data


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
    try:
        cuda = torch.cuda
        if hasattr(cuda, "manual_seed"):
            cuda.manual_seed(seed)
        if hasattr(cuda, "manual_seed_all"):
            cuda.manual_seed_all(seed)
    except Exception:
        pass


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


def _detect_enable_byt5(model_dir: str) -> bool:
    transformer_dir = Path(model_dir) / "transformer"
    index_files = sorted(transformer_dir.glob("*.safetensors.index.json"))
    for index_file in index_files:
        try:
            if "byt5_in." in index_file.read_text():
                return True
        except Exception:
            pass

    try:
        from safetensors import safe_open
    except Exception:
        return False

    safetensors_files = sorted(transformer_dir.glob("*.safetensors"))
    for safetensors_file in safetensors_files:
        if safetensors_file.name.endswith(".index.json"):
            continue
        try:
            with safe_open(str(safetensors_file), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k.startswith("byt5_in."):
                        return True
        except Exception:
            pass
    return False


def _build_pipeline(
    model_dir: str,
    vae_dir: str,
    vae_subfolder: str,
    pipe_dtype: torch.dtype,
    vae_dtype: torch.dtype,
    low_cpu_mem_usage: bool,
    enable_text_encoder_cpu_offload: bool,
    device_id: int = 0,
    vae_tiling: bool = True,
    vae_slicing: bool = True,
) -> "MammothModa25Pipeline":
    device = torch.device(f"cuda:{device_id}")

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
    if vae_slicing and hasattr(vae, "enable_slicing"):
        try:
            vae.enable_slicing()
        except Exception:
            pass
    if vae_tiling and hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
        except Exception:
            pass

    from src.transformer_mammothmoda import WanTransformer3DModel

    try:
        transformer_config = WanTransformer3DModel.load_config(
            model_dir, subfolder="transformer"
        )
        config_enable_byt5 = transformer_config.get("enable_byt5", False)
    except Exception:
        config_enable_byt5 = False

    weights_enable_byt5 = _detect_enable_byt5(model_dir)
    enable_byt5 = bool(weights_enable_byt5)
    if bool(config_enable_byt5) != enable_byt5:
        print(
            f"enable_byt5 mismatch: config={bool(config_enable_byt5)} checkpoint={enable_byt5}. Using checkpoint."
        )

    transformer = WanTransformer3DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=pipe_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        enable_byt5=enable_byt5,
    )

    pipe = MammothModa25Pipeline.from_pretrained(
        model_dir,
        vae=vae,
        transformer=transformer,
        torch_dtype=pipe_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    pipe.to(device)
    if enable_text_encoder_cpu_offload:
        pipe.enable_text_encoder_cpu_offload()
    return pipe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained",
    )
    parser.add_argument(
        "--vae_dir",
        type=str,
        default="pretrained/vae",
    )
    parser.add_argument("--vae_subfolder", type=str, default="vae")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--keep_qwen_quoted_text", action="store_true")
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--num_frames", type=int, default=93)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--context_guidance_scale", type=float, default=1.0)
    parser.add_argument("--context_cfg_range", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pipe_dtype", type=str, default="bf16")
    parser.add_argument("--vae_dtype", type=str, default="fp32")
    parser.add_argument(
        "--vae_tiling", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--vae_slicing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--low_cpu_mem_usage", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--disable_text_encoder_cpu_offload", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--origin_output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=7)
    parser.add_argument("--video", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)

    model_dir = os.path.expanduser(args.model_dir)
    vae_dir = (
        os.path.expanduser(args.vae_dir) if args.vae_dir is not None else model_dir
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    origin_output_path = (
        Path(args.origin_output) if (args.origin_output and args.video) else None
    )
    if origin_output_path is not None:
        origin_output_path.parent.mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)
    enable_offload = not bool(args.disable_text_encoder_cpu_offload)
    pipe = _build_pipeline(
        model_dir=model_dir,
        vae_dir=vae_dir,
        vae_subfolder=args.vae_subfolder,
        pipe_dtype=_parse_dtype(args.pipe_dtype),
        vae_dtype=_parse_dtype(args.vae_dtype),
        low_cpu_mem_usage=bool(args.low_cpu_mem_usage),
        enable_text_encoder_cpu_offload=enable_offload,
        device_id=args.device_id,
        vae_tiling=bool(args.vae_tiling),
        vae_slicing=bool(args.vae_slicing),
    )

    video = None
    if args.video:
        video_tensor = open_video(
            args.video,
            0,
            args.num_frames,
            target_height=args.height,
            target_width=args.width,
        )
        video = [video_tensor]
        args.height = video_tensor.shape[2]
        args.width = video_tensor.shape[3]
    else:
        if args.height <= 0:
            args.height = 480
        if args.width <= 0:
            args.width = 832

    if origin_output_path is not None and video is not None:
        out_video, origin_video = pipe(
            video=video,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            keep_qwen_quoted_text=bool(args.keep_qwen_quoted_text),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            context_guidance_scale=args.context_guidance_scale,
            context_cfg_range=tuple(args.context_cfg_range),
            return_dict=False,
            return_origin_video=True,
        )
        output_frames = out_video[0]
        origin_frames = origin_video[0] if origin_video is not None else None
    else:
        output_frames = pipe(
            video=video,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            keep_qwen_quoted_text=bool(args.keep_qwen_quoted_text),
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            context_guidance_scale=args.context_guidance_scale,
            context_cfg_range=tuple(args.context_cfg_range),
        ).frames[0]
        origin_frames = None

    suffix = output_path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        frame = output_frames[0]
        if isinstance(frame, Image.Image):
            image = frame
        else:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu()
                if frame.dim() == 3 and frame.shape[0] in (1, 3):
                    frame = frame.permute(1, 2, 0)
                frame = frame.float()
                frame_min = float(frame.min())
                frame_max = float(frame.max())
                if frame_min < 0.0:
                    frame = (frame + 1.0) / 2.0
                    frame_min = float(frame.min())
                    frame_max = float(frame.max())
                if frame_max > 1.0 or frame_min < 0.0:
                    frame = frame.clamp(0.0, 1.0)
                frame = (frame * 255.0).round().to(torch.uint8).numpy()
            else:
                import numpy as np

                if not isinstance(frame, np.ndarray):
                    frame = np.array(frame)
                if frame.dtype != np.uint8:
                    frame_min = float(frame.min())
                    frame_max = float(frame.max())
                    if frame_min < 0.0:
                        frame = (frame + 1.0) / 2.0
                        frame_min = float(frame.min())
                        frame_max = float(frame.max())
                    if frame_max > 1.0 or frame_min < 0.0:
                        frame = np.clip(frame, 0.0, 1.0)
                    frame = (frame * 255.0).round().astype("uint8")
            image = Image.fromarray(frame)
        image.save(str(output_path))
        if (
            origin_output_path is not None
            and origin_frames is not None
            and len(origin_frames) > 0
        ):
            origin_frame = origin_frames[0]
            if isinstance(origin_frame, Image.Image):
                origin_image = origin_frame
            else:
                if isinstance(origin_frame, torch.Tensor):
                    origin_frame = origin_frame.detach().cpu()
                    if origin_frame.dim() == 3 and origin_frame.shape[0] in (1, 3):
                        origin_frame = origin_frame.permute(1, 2, 0)
                    origin_frame = origin_frame.float()
                    frame_min = float(origin_frame.min())
                    frame_max = float(origin_frame.max())
                    if frame_min < 0.0:
                        origin_frame = (origin_frame + 1.0) / 2.0
                        frame_min = float(origin_frame.min())
                        frame_max = float(origin_frame.max())
                    if frame_max > 1.0 or frame_min < 0.0:
                        origin_frame = origin_frame.clamp(0.0, 1.0)
                    origin_frame = (
                        (origin_frame * 255.0).round().to(torch.uint8).numpy()
                    )
                else:
                    import numpy as np

                    if not isinstance(origin_frame, np.ndarray):
                        origin_frame = np.array(origin_frame)
                    if origin_frame.dtype != np.uint8:
                        frame_min = float(origin_frame.min())
                        frame_max = float(origin_frame.max())
                        if frame_min < 0.0:
                            origin_frame = (origin_frame + 1.0) / 2.0
                            frame_min = float(origin_frame.min())
                            frame_max = float(origin_frame.max())
                        if frame_max > 1.0 or frame_min < 0.0:
                            origin_frame = np.clip(origin_frame, 0.0, 1.0)
                        origin_frame = (origin_frame * 255.0).round().astype("uint8")
                origin_image = Image.fromarray(origin_frame)
            origin_image.save(str(origin_output_path))
    else:
        from diffusers.utils import export_to_video

        export_to_video(
            output_frames,
            str(output_path),
            quality=int(args.quality),
            fps=int(args.fps),
        )
        if origin_output_path is not None and origin_frames is not None:
            export_to_video(
                origin_frames,
                str(origin_output_path),
                quality=int(args.quality),
                fps=int(args.fps),
            )


if __name__ == "__main__":
    main()
