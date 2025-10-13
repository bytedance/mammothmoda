# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from diffusers.utils.torch_utils import randn_tensor

from mammothmoda2.model.mammothmoda2_dit import FlowMatchEulerDiscreteScheduler, RotaryPosEmbedReal

from .misc import Singleton
from .vae_processor import Mammothmoda2VAEImageProcessor

if TYPE_CHECKING:
    import numpy as np
    from diffusers.image_processor import PipelineImageInput
    from PIL import Image

    from mammothmoda2.model import Mammothmoda2Model, MammothUTokenizer

__all__ = ["decode_diffusion_image"]


class VAEPostProcessor(Singleton):
    def __init__(self, vae_scale_factor=16, max_side_length=2048) -> None:
        if self._initialized:
            return
        self._vae_transform = Mammothmoda2VAEImageProcessor(
            vae_scale_factor=vae_scale_factor,
            max_side_length=max_side_length,
        )
        self._initialized = True

    def __call__(self, image, output_type="pil") -> "Image.Image | np.ndarray | torch.Tensor":
        return self._vae_transform.postprocess(image, output_type=output_type)


def remap_unified_tokens(generated_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Remaps generated string-based visual tokens back to integer token IDs."""
    unified_tokens_str = tokenizer.decode(generated_ids)
    visual_template_regex = r"<\|visual token (\d+)\|>"
    token_ids = re.findall(visual_template_regex, unified_tokens_str)
    return torch.tensor([int(m) for m in token_ids], dtype=torch.long)


def extract_condition_tokens(
    hidden_states: torch.FloatTensor,
    condition_token_mask: torch.BoolTensor,
) -> tuple:
    B, L, D = hidden_states.shape
    pos = torch.arange(L, device=hidden_states.device).unsqueeze(0).expand(B, -1)  # 0..L-1
    # mask=True, ~mask=0, key from 0..L-1
    # mask=False, ~mask=1, key from L..2L-1
    key: torch.LongTensor = (~condition_token_mask).to(torch.long) * L + pos
    # place all the condition tokens in front
    condition_sorted_indices = key.argsort(dim=1, stable=True)
    idx = condition_sorted_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, L, D)
    condition_tokens_sorted = torch.gather(hidden_states, dim=1, index=idx)  # (B, L, D)
    condition_lengths = condition_token_mask.sum(dim=1)  # (B,)
    max_condition_len = int(condition_lengths.max().item())
    if max_condition_len == 0:
        return hidden_states.new_zeros(B, 0, D), hidden_states.new_zeros(B, 0, dtype=torch.bool)

    condition_tokens_compact = condition_tokens_sorted[:, :max_condition_len, :]  # (B, max_condition_len, D)
    condition_attention_mask = torch.arange(max_condition_len, device=hidden_states.device).expand(
        B, -1
    ) < condition_lengths.unsqueeze(1)
    return condition_tokens_compact, condition_attention_mask


@torch.inference_mode()
def encode_full_prompts(
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    negative_ids: torch.LongTensor,
    negative_mask: torch.BoolTensor,
    questions_mask: torch.BoolTensor | None = None,
    answers_mask: torch.BoolTensor | None = None,
    **kwargs,  # noqa: ARG001
) -> tuple:
    # 1. convert input_ids to positive_hidden_states
    output = model.llm_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_last_hidden_states=True,
    )
    positive_hidden_states = output.last_hidden_states
    B, L, D = positive_hidden_states.shape
    gen_token_mask = input_ids >= model.llm_model.config.gen_vocab_start_index
    if "text" in model.config.gen_condition_mode:
        visual_tokens_ids = tokenizer.visual_tokens_ids
        visual_token_mask = torch.isin(
            input_ids, torch.tensor(visual_tokens_ids, dtype=torch.long, device=input_ids.device)
        )
        # all the text tokens in question
        text_condition_token_mask = questions_mask & ~(visual_token_mask | gen_token_mask) & attention_mask
        text_condition_tokens_compact, text_condition_attention_mask = extract_condition_tokens(
            positive_hidden_states, text_condition_token_mask
        )
    else:
        text_condition_tokens_compact = positive_hidden_states.new_zeros(B, 0, D)
        text_condition_attention_mask = positive_hidden_states.new_zeros(B, 0, dtype=torch.bool)

    if "image" in model.config.gen_condition_mode:
        # all the gen image tokens in answer
        image_condition_token_mask = answers_mask & gen_token_mask
        image_condition_tokens_compact, image_condition_attention_mask = extract_condition_tokens(
            positive_hidden_states, image_condition_token_mask
        )
        if model.gen_image_condition_refiner is not None:
            image_condition_tokens_compact = model.gen_image_condition_refiner(
                image_condition_tokens_compact, ~image_condition_attention_mask.bool()
            )
            image_condition_attention_mask = torch.ones(
                image_condition_tokens_compact.shape[:2], dtype=torch.bool, device=image_condition_tokens_compact.device
            )
    else:
        image_condition_tokens_compact = positive_hidden_states.new_zeros(B, 0, D)
        image_condition_attention_mask = positive_hidden_states.new_zeros(B, 0, dtype=torch.bool)

    condition_tokens_compact = torch.cat([text_condition_tokens_compact, image_condition_tokens_compact], dim=1)
    condition_attention_mask = torch.cat([text_condition_attention_mask, image_condition_attention_mask], dim=1)

    # 2. convert negative_ids to negative_hidden_states
    if negative_ids is not None:
        output = model.llm_model(
            input_ids=negative_ids,
            attention_mask=negative_mask,
            output_last_hidden_states=True,
        )
        negative_condition_tokens = output.last_hidden_states
        negative_attention_mask = negative_mask
    else:
        negative_condition_tokens = None
        negative_attention_mask = None

    return (
        condition_tokens_compact,
        condition_attention_mask,
        negative_condition_tokens,
        negative_attention_mask,
    )


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    **kwargs,
) -> tuple:
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            error_msg = (
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
            raise ValueError(error_msg)
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@torch.inference_mode()
def processing(
    latents,
    ref_latents,
    scheduler: FlowMatchEulerDiscreteScheduler,
    model,
    prompt_embeds,
    prompt_attention_mask,
    negative_prompt_embeds,
    negative_attention_mask,
    freqs_cis,
    num_inference_steps,
    device,
    dtype,
    cfg_range=(0.0, 1.0),
    text_guidance_scale=1.0,
    image_guidance_scale=1.0,
) -> "PipelineImageInput":
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, timesteps=None, num_tokens=latents.shape[-2] * latents.shape[-1]
    )

    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        model_pred = model.gen_transformer(
            hidden_states=latents,
            timestep=timestep,
            text_hidden_states=prompt_embeds,
            text_attention_mask=prompt_attention_mask,
            ref_image_hidden_states=ref_latents,
            freqs_cis=freqs_cis,
        )
        text_guidance_scale = text_guidance_scale if cfg_range[0] <= i / len(timesteps) <= cfg_range[1] else 1.0
        image_guidance_scale = image_guidance_scale if cfg_range[0] <= i / len(timesteps) <= cfg_range[1] else 1.0

        if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
            model_pred_ref = model.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=negative_prompt_embeds,
                text_attention_mask=negative_attention_mask,
                ref_image_hidden_states=ref_latents,
                freqs_cis=freqs_cis,
            )

            model_pred_uncond = model.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=negative_prompt_embeds,
                text_attention_mask=negative_attention_mask,
                ref_image_hidden_states=None,
                freqs_cis=freqs_cis,
            )

            model_pred = (
                model_pred_uncond
                + image_guidance_scale * (model_pred_ref - model_pred_uncond)
                + text_guidance_scale * (model_pred - model_pred_ref)
            )

        elif text_guidance_scale > 1.0:
            model_pred_uncond = model.gen_transformer(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=negative_prompt_embeds,
                text_attention_mask=negative_attention_mask,
                ref_image_hidden_states=None,
                freqs_cis=freqs_cis,
            )

            model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

        latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]
        latents = latents.to(dtype=dtype)

    latents = latents.to(dtype=dtype)
    if model.gen_vae.config.scaling_factor is not None:
        latents = latents / model.gen_vae.config.scaling_factor
    if model.gen_vae.config.shift_factor is not None:
        latents = latents + model.gen_vae.config.shift_factor
    image = model.gen_vae.decode(latents, return_dict=False)[0]
    return image


@torch.inference_mode()
def decode_diffusion_image(
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    negative_ids: torch.Tensor | None,
    negative_mask: torch.Tensor | None,
    model: "Mammothmoda2Model",
    tokenizer: "MammothUTokenizer",
    output_dir: str,
    num_images_per_prompt: int = 1,
    text_guidance_scale: float = 5.0,
    vae_scale_factor: float = 16,
    cfg_range: tuple[float, float] = (0.0, 1.0),
    num_inference_steps: int = 50,
    height: int = 512,
    width: int = 512,
    device: torch.device = None,
    index: int | None = 0,
    **kwargs,  # noqa: ARG001
) -> dict:
    # 1. Get the full condition_features, both positive and negative
    device = device or next(model.parameters()).device
    input_ids = generated_ids[:num_images_per_prompt]
    img_end_id = tokenizer.get_vocab()[tokenizer.eoi_token]  # <|image end|>
    img_end = torch.full((num_images_per_prompt, 1), img_end_id, dtype=torch.int64, device=device)
    input_ids = torch.concat([input_ids, img_end], dim=1)
    attention_mask = attention_mask[:num_images_per_prompt]
    attention_mask = torch.concat(
        [attention_mask, torch.ones((num_images_per_prompt, 1), dtype=torch.int64, device=device)], dim=1
    )
    answer_start_index = input_ids.shape[1] - 10
    questions_mask = attention_mask.clone()
    questions_mask[:, answer_start_index:] = 0
    answers_mask = attention_mask.clone()
    answers_mask[:, :answer_start_index] = 0
    (
        condition_tokens_compact,
        condition_attention_mask,
        negative_condition_tokens,
        negative_attention_mask,
    ) = encode_full_prompts(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        negative_ids=negative_ids,
        negative_mask=negative_mask,
        questions_mask=questions_mask,
        answers_mask=answers_mask,
    )

    # 2. Prepare reference latents for control image
    ref_latents = []
    image_guidance_scale = 1
    ref_latents = None
    h, w = height, width
    ref_latents = [ref_latents] * num_images_per_prompt

    # 3. Prepare input latents for generated image
    latent_channels = model.gen_transformer.config.in_channels
    shape = (num_images_per_prompt, latent_channels, 2 * h // vae_scale_factor, 2 * w // vae_scale_factor)
    latents = randn_tensor(shape, device=device, dtype=condition_tokens_compact.dtype)

    freqs_cis = RotaryPosEmbedReal.get_freqs_real(
        model.gen_transformer.config.axes_dim_rope,
        model.gen_transformer.config.axes_lens,
        theta=10000,
    )

    # 4. Generate the image by diffusion model
    scheduler = FlowMatchEulerDiscreteScheduler()
    image = processing(
        latents,
        ref_latents,
        scheduler,
        model,
        prompt_embeds=condition_tokens_compact,
        prompt_attention_mask=condition_attention_mask,
        negative_prompt_embeds=negative_condition_tokens,
        negative_attention_mask=negative_attention_mask,
        freqs_cis=freqs_cis,
        num_inference_steps=num_inference_steps,
        device=device,
        dtype=condition_tokens_compact.dtype,
        cfg_range=cfg_range,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
    )

    # 5. Save the generated image
    images = VAEPostProcessor()(image)
    return_info = []
    if not Path(output_dir).is_dir():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    for bs in range(num_images_per_prompt):
        save_path = f"{output_dir}/image{bs}.png" if index is None else f"{output_dir}/image_index{index}_bs{bs}.png"
        images[bs].save(save_path)
        return_info.append({"save_path": save_path, "decoded_img": images[bs]})
    return return_info
