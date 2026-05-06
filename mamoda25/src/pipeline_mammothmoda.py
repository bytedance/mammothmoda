import html
import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ftfy
import torch
from diffusers import WanPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import (
    AutoTokenizer,
    ByT5Tokenizer,
    Qwen3VLForConditionalGeneration,
    T5EncoderModel,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


QUOTED_CONTENT_PATTERN = r"\"(.*?)\"|“(.*?)”|(?<!\w)'(.*?)(?<!\s)'(?!\w)|‘(.*?)’|「(.*?)」|『(.*?)』|《(.*?)》"


def extract_glyph_texts(prompt: str) -> Optional[str]:
    """
    Extract glyph texts from prompt using regex pattern.

    Args:
        prompt: Input prompt string

    Returns:
        List of extracted glyph texts
    """
    matches = re.findall(QUOTED_CONTENT_PATTERN, prompt)
    result = [next((g for g in match if g), "") for match in matches]
    result = list(dict.fromkeys(result)) if len(result) > 1 else result

    if result:
        formatted_result = ". ".join([f'Text "{text}"' for text in result]) + ". "
    else:
        formatted_result = None

    return formatted_result


def remove_quoted_content(text: Union[str, list[str]]) -> Union[str, list[str]]:
    def _replace_func(match: re.Match) -> str:
        matched_str = match.group()
        if matched_str.startswith('"'):
            return '""'
        if matched_str.startswith("“"):
            return "“”"
        if matched_str.startswith("'"):
            return "''"
        if matched_str.startswith("‘"):
            return "‘’"
        if matched_str.startswith("「"):
            return "「」"
        if matched_str.startswith("『"):
            return "『』"
        if matched_str.startswith("《"):
            return "《》"
        return matched_str

    if isinstance(text, str):
        return re.sub(QUOTED_CONTENT_PATTERN, _replace_func, text)
    elif isinstance(text, list) and all(isinstance(s, str) for s in text):
        return [re.sub(QUOTED_CONTENT_PATTERN, _replace_func, s) for s in text]
    else:
        return text


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def _get_byt5_prompt_embeds(
    tokenizer: ByT5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    device: torch.device,
    tokenizer_max_length: int = 256,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    glyph_texts = [extract_glyph_texts(p) for p in prompt]
    logger.debug("glyph_texts: %s", glyph_texts)
    prompt_embeds_list = []
    prompt_embeds_mask_list = []

    for glyph_text in glyph_texts:
        if glyph_text is None:
            glyph_text_embeds = torch.zeros(
                (1, tokenizer_max_length, text_encoder.config.d_model),
                device=device,
                dtype=text_encoder.dtype,
            )
            glyph_text_embeds_mask = torch.zeros(
                (1, tokenizer_max_length), device=device, dtype=torch.int64
            )
        else:
            txt_tokens = tokenizer(
                glyph_text,
                padding="max_length",
                max_length=tokenizer_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(device)

            glyph_text_embeds = text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask.float(),
            )[0]
            glyph_text_embeds = glyph_text_embeds.to(device=device)
            glyph_text_embeds_mask = txt_tokens.attention_mask.to(device=device)

        prompt_embeds_list.append(glyph_text_embeds)
        prompt_embeds_mask_list.append(glyph_text_embeds_mask)

    prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
    prompt_embeds_mask = torch.cat(prompt_embeds_mask_list, dim=0)

    return prompt_embeds, prompt_embeds_mask


class MammothModa25Pipeline(DiffusionPipeline, WanLoraLoaderMixin):

    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    @property
    def _execution_device(self):
        if getattr(self, "_text_encoder_cpu_offload", False):
            for name in ("transformer", "transformer_2", "vae"):
                module = getattr(self, name, None)
                if module is None or not isinstance(module, torch.nn.Module):
                    continue
                try:
                    return next(module.parameters()).device
                except StopIteration:
                    try:
                        return next(module.buffers()).device
                    except StopIteration:
                        pass
        return super()._execution_device

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: Qwen3VLForConditionalGeneration,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: ByT5Tokenizer,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
            tokenizer_2=tokenizer_2,
            text_encoder_2=text_encoder_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = (
            self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )

        # Use qwen3vl
        self.text_encoder = self.text_encoder.model
        self._text_encoder_cpu_offload = False
        self.tokenizer_2_max_length = 256

    def enable_text_encoder_cpu_offload(self):
        self._text_encoder_cpu_offload = True
        if getattr(self, "text_encoder", None) is not None:
            self.text_encoder.to(torch.device("cpu"))
        if getattr(self, "text_encoder_2", None) is not None:
            self.text_encoder_2.to(torch.device("cpu"))
        return self

    def disable_text_encoder_cpu_offload(self, device: Optional[torch.device] = None):
        self._text_encoder_cpu_offload = False
        target_device = (
            device or getattr(self, "_execution_device", None) or torch.device("cpu")
        )
        if getattr(self, "text_encoder", None) is not None:
            self.text_encoder.to(target_device)
        if getattr(self, "text_encoder_2", None) is not None:
            self.text_encoder_2.to(target_device)
        return self

    def to(self, *args, **kwargs):
        pipe = super().to(*args, **kwargs)
        if (
            getattr(pipe, "_text_encoder_cpu_offload", False)
            and getattr(pipe, "text_encoder", None) is not None
        ):
            pipe.text_encoder.to(torch.device("cpu"))
            if getattr(pipe, "text_encoder_2", None) is not None:
                pipe.text_encoder_2.to(torch.device("cpu"))
        return pipe

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_byt5: bool = False,
        keep_qwen_quoted_text: bool = False,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        if use_byt5 and not keep_qwen_quoted_text:
            prompt = remove_quoted_content(prompt)
        logger.debug("prompt: %s", prompt)
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )
        mask = mask.to(device)
        if num_videos_per_prompt != 1:
            mask = mask.repeat_interleave(num_videos_per_prompt, dim=0)
        return prompt_embeds, mask

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        keep_qwen_quoted_text: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        execution_device = device or self._execution_device
        encoder_device = (
            torch.device("cpu")
            if getattr(self, "_text_encoder_cpu_offload", False)
            else execution_device
        )

        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        if prompt_list is not None:
            batch_size = len(prompt_list)
        else:
            batch_size = prompt_embeds.shape[0]

        enable_byt5 = bool(
            getattr(self, "tokenizer_2", None) is not None
            and getattr(self, "text_encoder_2", None) is not None
            and getattr(getattr(self, "transformer", None), "config", None) is not None
            and getattr(self.transformer.config, "enable_byt5", False)
        )

        prompt_embeds_mask = None
        negative_prompt_embeds_mask = None
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_t5_prompt_embeds(
                prompt=prompt_list,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=encoder_device,
                dtype=dtype,
                use_byt5=enable_byt5,
                keep_qwen_quoted_text=keep_qwen_quoted_text,
            )
        else:
            prompt_embeds_mask = torch.ones(
                (prompt_embeds.shape[0], prompt_embeds.shape[1]),
                device=prompt_embeds.device,
                dtype=torch.int64,
            )

        # byt5 prompt embeds
        prompt_embeds_2 = None
        prompt_embeds_mask_2 = None
        negative_prompt_embeds_2 = None
        negative_prompt_embeds_mask_2 = None

        if enable_byt5:
            prompt_embeds_2, prompt_embeds_mask_2 = _get_byt5_prompt_embeds(
                tokenizer=self.tokenizer_2,
                text_encoder=self.text_encoder_2,
                prompt=prompt,
                device=encoder_device,
                tokenizer_max_length=self.tokenizer_2_max_length,
            )
            if num_videos_per_prompt != 1:
                prompt_embeds_2 = prompt_embeds_2.repeat_interleave(
                    num_videos_per_prompt, dim=0
                )
                prompt_embeds_mask_2 = prompt_embeds_mask_2.repeat_interleave(
                    num_videos_per_prompt, dim=0
                )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_list = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt_list is not None and type(prompt_list) is not type(
                negative_prompt_list
            ):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt_list)} !="
                    f" {type(prompt_list)}."
                )
            elif batch_size != len(negative_prompt_list):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt_list} has batch size {len(negative_prompt_list)}, but `prompt`:"
                    f" {prompt_list} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_embeds_mask = (
                self._get_t5_prompt_embeds(
                    prompt=negative_prompt_list,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=encoder_device,
                    dtype=dtype,
                    use_byt5=enable_byt5,
                    keep_qwen_quoted_text=keep_qwen_quoted_text,
                )
            )
            # byt5 negative prompt embeds
            if enable_byt5:
                negative_prompt_embeds_2, negative_prompt_embeds_mask_2 = (
                    _get_byt5_prompt_embeds(
                        tokenizer=self.tokenizer_2,
                        text_encoder=self.text_encoder_2,
                        prompt=negative_prompt_list,
                        device=encoder_device,
                        tokenizer_max_length=self.tokenizer_2_max_length,
                    )
                )
                if num_videos_per_prompt != 1:
                    negative_prompt_embeds_2 = (
                        negative_prompt_embeds_2.repeat_interleave(
                            num_videos_per_prompt, dim=0
                        )
                    )
                    negative_prompt_embeds_mask_2 = (
                        negative_prompt_embeds_mask_2.repeat_interleave(
                            num_videos_per_prompt, dim=0
                        )
                    )
        elif do_classifier_free_guidance and negative_prompt_embeds is not None:
            negative_prompt_embeds_mask = torch.ones(
                (negative_prompt_embeds.shape[0], negative_prompt_embeds.shape[1]),
                device=negative_prompt_embeds.device,
                dtype=torch.int64,
            )

        output_dtype = (
            dtype
            or getattr(getattr(self, "transformer", None), "dtype", None)
            or getattr(getattr(self, "transformer_2", None), "dtype", None)
            or getattr(getattr(self, "text_encoder", None), "dtype", None)
        )

        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(
                device=execution_device, dtype=output_dtype
            )
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(device=execution_device)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=execution_device, dtype=output_dtype
            )
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(
                device=execution_device, dtype=output_dtype
            )
        if negative_prompt_embeds_mask is not None:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(
                device=execution_device
            )
        if negative_prompt_embeds_2 is not None:
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(
                device=execution_device, dtype=output_dtype
            )
        if prompt_embeds_mask_2 is not None:
            prompt_embeds_mask_2 = prompt_embeds_mask_2.to(device=execution_device)
        if negative_prompt_embeds_mask_2 is not None:
            negative_prompt_embeds_mask_2 = negative_prompt_embeds_mask_2.to(
                device=execution_device
            )

        return (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
        )

    @torch.no_grad()
    def __call__(
        self,
        video: Optional[List[torch.Tensor]] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        context_guidance_scale: float = 1.0,
        context_cfg_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        return_origin_video: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        keep_qwen_quoted_text: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        def _nearest_multiple(value: int, multiple: int) -> int:
            value = int(value)
            if value <= 0:
                return multiple
            lower = (value // multiple) * multiple
            upper = lower + multiple
            if lower == 0:
                lower = multiple
            if value - lower <= upper - value:
                return lower
            return upper

        height = _nearest_multiple(height, 32)
        width = _nearest_multiple(width, 32)
        self.height = height
        self.width = width

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames
                // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            prompt_embeds_2,
            prompt_embeds_mask_2,
            negative_prompt_embeds_2,
            negative_prompt_embeds_mask_2,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            keep_qwen_quoted_text=keep_qwen_quoted_text,
            device=device,
        )

        transformer_dtype = (
            self.transformer.dtype
            if self.transformer is not None
            else self.transformer_2.dtype
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds_2 is not None:
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(transformer_dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )

        video_vae_features = None
        if video is not None:
            latents, video_vae_features = self.prepare_video_latents(
                batch_size, video, prompt_embeds.device, torch.float32, generator
            )
        else:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = (
                self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
            )
        else:
            boundary_timestep = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # wan2.1 or high-noise stage in wan2.2
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    # low-noise stage in wan2.2
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    # batch_size, seq_len
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        video_vae_features=video_vae_features,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        byt5_prompt_embeds=prompt_embeds_2,
                        prompt_mask=prompt_embeds_mask,
                        byt5_prompt_embeds_mask=prompt_embeds_mask_2,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    text_scale = current_guidance_scale
                    context_scale = 1.0
                    if video_vae_features is not None and context_guidance_scale > 1.0:
                        if context_cfg_range is None:
                            context_in_range = True
                        else:
                            step_ratio = i / len(timesteps)
                            context_in_range = (
                                context_cfg_range[0]
                                <= step_ratio
                                <= context_cfg_range[1]
                            )
                        if context_in_range:
                            context_scale = context_guidance_scale

                    if text_scale > 1.0 or context_scale > 1.0:
                        if context_scale > 1.0:
                            with current_model.cache_context("context"):
                                noise_context = current_model(
                                    hidden_states=latent_model_input,
                                    video_vae_features=video_vae_features,
                                    timestep=timestep,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]
                            with current_model.cache_context("uncond"):
                                noise_uncond = current_model(
                                    hidden_states=latent_model_input,
                                    video_vae_features=None,
                                    timestep=timestep,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    attention_kwargs=attention_kwargs,
                                    return_dict=False,
                                )[0]
                            if text_scale > 1.0:
                                noise_pred = (
                                    noise_uncond
                                    + context_scale * (noise_context - noise_uncond)
                                    + text_scale * (noise_pred - noise_context)
                                )
                            else:
                                noise_pred = noise_uncond + context_scale * (
                                    noise_context - noise_uncond
                                )
                        else:
                            with current_model.cache_context("uncond"):
                                noise_uncond = current_model(
                                    hidden_states=latent_model_input,
                                    video_vae_features=video_vae_features,
                                    timestep=timestep,
                                    encoder_hidden_states=negative_prompt_embeds,
                                    attention_kwargs=attention_kwargs,
                                    byt5_prompt_embeds=negative_prompt_embeds_2,
                                    prompt_mask=negative_prompt_embeds_mask,
                                    byt5_prompt_embeds_mask=negative_prompt_embeds_mask_2,
                                    return_dict=False,
                                )[0]
                            noise_pred = noise_uncond + text_scale * (
                                noise_pred - noise_uncond
                            )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )
            origin_video = None
            if return_origin_video and video_vae_features is not None:
                origin_latents = video_vae_features.to(self.vae.dtype)
                origin_latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(origin_latents.device, origin_latents.dtype)
                )
                origin_latents_std = 1.0 / torch.tensor(
                    self.vae.config.latents_std
                ).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                    origin_latents.device, origin_latents.dtype
                )
                origin_latents = (
                    origin_latents / origin_latents_std + origin_latents_mean
                )
                origin_decoded = self.vae.decode(origin_latents, return_dict=False)[0]
                origin_video = self.video_processor.postprocess_video(
                    origin_decoded, output_type=output_type
                )
        else:
            video = latents
            origin_video = video_vae_features if return_origin_video else None

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            if return_origin_video:
                return (video, origin_video)
            return (video,)

        return WanPipelineOutput(frames=video)

    def prepare_video_latents(self, batch_size, video, device, dtype, generator=None):
        self.patch_size = [1, 2, 2]
        # process video data
        video = torch.stack(video, dim=0)
        video = video.to(torch.float32) / 255.0  # normalize to [0, 1]
        video = video.permute(0, 2, 1, 3, 4)
        h, w = video.shape[-2:]

        max_area = self.height * self.width
        aspect_ratio = h / w
        latent_h = round(
            math.sqrt(max_area * aspect_ratio)
            // self.vae_scale_factor_spatial
            // self.patch_size[1]
            // 1
            * self.patch_size[1]
            * 1
        )
        latent_w = round(
            math.sqrt(max_area / aspect_ratio)
            // self.vae_scale_factor_spatial
            // self.patch_size[2]
            // 1
            * self.patch_size[2]
            * 1
        )

        h = latent_h * self.vae_scale_factor_spatial
        w = latent_w * self.vae_scale_factor_spatial
        num_latent_frames = (video.size(2) - 1) // self.vae_scale_factor_temporal + 1

        b, c, t, orig_h, orig_w = video.shape
        video_4d = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, orig_h, orig_w)
        video_4d = torch.nn.functional.interpolate(
            video_4d, size=(h, w), mode="bilinear", align_corners=False
        )
        video = video_4d.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).to(device)

        video = 2.0 * video - 1.0
        vae_feature = [
            retrieve_latents(self.vae.encode(vid.unsqueeze(0)), sample_mode="argmax")
            for vid in video
        ]
        vae_feature = torch.cat(vae_feature, dim=0).to(dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, dtype)
        vae_feature = (vae_feature - latents_mean) * latents_std

        noise = self.prepare_latents(
            batch_size,
            48,
            h,
            w,
            video.size(2),
            torch.float32,
            device,
            generator,
            None,
        )
        return noise, vae_feature

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}"
            )

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError(
                "`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs
