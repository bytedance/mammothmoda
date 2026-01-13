from typing import Optional
import torch
from diffusers import WanPipeline
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel 
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class MammothModa25Pipeline(WanPipeline):

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
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )
        # Use qwen3vl 
        self.text_encoder = self.text_encoder.model
        self._text_encoder_cpu_offload = False

    def enable_text_encoder_cpu_offload(self):
        self._text_encoder_cpu_offload = True
        if getattr(self, "text_encoder", None) is not None:
            self.text_encoder.to(torch.device("cpu"))
        return self

    def disable_text_encoder_cpu_offload(self, device: Optional[torch.device] = None):
        self._text_encoder_cpu_offload = False
        if getattr(self, "text_encoder", None) is not None:
            target_device = device or getattr(self, "_execution_device", None) or torch.device("cpu")
            self.text_encoder.to(target_device)
        return self

    def to(self, *args, **kwargs):
        pipe = super().to(*args, **kwargs)
        if getattr(pipe, "_text_encoder_cpu_offload", False) and getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder.to(torch.device("cpu"))
        return pipe

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        execution_device = device or self._execution_device
        encoder_device = torch.device("cpu") if getattr(self, "_text_encoder_cpu_offload", False) else execution_device

        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        if prompt_list is not None:
            batch_size = len(prompt_list)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_list,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=encoder_device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_list = (
                batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            )

            if prompt_list is not None and type(prompt_list) is not type(negative_prompt_list):
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

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt_list,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=encoder_device,
                dtype=dtype,
            )

        output_dtype = (
            dtype
            or getattr(getattr(self, "transformer", None), "dtype", None)
            or getattr(getattr(self, "transformer_2", None), "dtype", None)
            or getattr(getattr(self, "text_encoder", None), "dtype", None)
        )

        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.to(device=execution_device, dtype=output_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(device=execution_device, dtype=output_dtype)

        return prompt_embeds, negative_prompt_embeds
