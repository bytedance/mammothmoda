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

import os
from typing import TYPE_CHECKING, ClassVar

import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger
from torch import nn
from transformers import GenerationConfig, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.generation import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from mammothmoda2.utils import ClassifierFreeGuidanceLogitsProcessor, SampledScopeLogitsProcessor

from .configuration_mammothmoda2 import Mammothmoda2Config
from .mammothmoda2_dit import (
    RotaryPosEmbedReal,
    SimpleQFormerImageRefiner,
    Transformer2DModel,
    create_transport,
)
from .mammothmoda2_qwen2_5_vl.modeling_mammothmoda2_qwen2_5_vl import (
    Mammothmoda2Qwen2_5_VLCausalLMOutputWithPast,
    Mammothmoda2Qwen2_5_VLForConditionalGeneration,
    Qwen2RMSNorm,
)

if TYPE_CHECKING:
    from transformers.generation.utils import GenerateOutput
    from transformers.modeling_utils import SpecificPreTrainedModelType


class Mammothmoda2PreTrainedModel(PreTrainedModel):
    """Mammothmoda VL Pretrained model inherit from Qwen2PreTrainedModel."""

    config_class = Mammothmoda2Config
    base_model_prefix = "llm_model"  # Key to the llm base model for all HF style nested PreTrainedModel.
    supports_gradient_checkpointing = True
    _skip_keys_device_placement: ClassVar = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def __init__(self, config: Mammothmoda2Config, *inputs, **kwargs) -> None:
        """Initialize the Mammothmoda2PreTrainedModel."""
        super().__init__(config, *inputs, **kwargs)
        self.config = config  # Redundant assignment For better hinting.
        self.t2i_generate = False
        self.generation_config = GenerationConfig.from_model_config(config)

    def _init_weights(self, module) -> None:
        """Initialize the Mammothmoda2Model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(
        cls: "SpecificPreTrainedModelType",
        pretrained_model_name_or_path: str | os.PathLike | None,
        t2i_generate: bool = False,
        *args,
        **kwargs,
    ) -> "SpecificPreTrainedModelType":
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model.t2i_generate = t2i_generate
        try:
            model.generation_config = GenerationConfig.from_pretrained(
                pretrained_model_name_or_path, config_file_name=model.generation_config_file_name
            )
        except Exception as e:
            error_msg = f"Failed to load generation_config.json from {pretrained_model_name_or_path} with error {e}."
            logger.error(error_msg)
        return model

    def save_pretrained(self, save_directory: str | os.PathLike, **kwargs) -> None:
        super().save_pretrained(save_directory, **kwargs)
        self.generation_config.save_pretrained(save_directory, config_file_name=self.generation_config_file_name)

    @property
    def generation_config_file_name(self) -> str:
        return "t2i_generation_config.json" if self.t2i_generate else "und_generation_config.json"


class Mammothmoda2Model(Mammothmoda2PreTrainedModel):
    """Mammothmoda combine Qwen2 model."""

    def __init__(self, config: Mammothmoda2Config, *inputs, **kwargs) -> None:
        """MammothmodaVL model initialization."""
        super().__init__(config)
        self.config = config  # Redundant assignment For better hinting.
        logger.info(f"=> Mammothmoda2Model config: {self.config}")

        # MLLM initialization
        self.llm_model = Mammothmoda2Qwen2_5_VLForConditionalGeneration._from_config(config=self.config.llm_config)  # noqa: SLF001

        # DiT initialization.
        self.gen_vae = AutoencoderKL.from_config(self.config.gen_vae_config)
        self.gen_transformer = Transformer2DModel.from_config(self.config.gen_dit_config)
        self.reinit_caption_embedder(self.llm_model.config.hidden_size)
        self.gen_freqs_cis = RotaryPosEmbedReal.get_freqs_real(
            config.gen_axes_dim_rope,
            config.gen_axes_lens,
            theta=10000,
        )
        self.gen_transport = create_transport(
            path_type="Linear",
            prediction="velocity",
            **config.gen_transport_config,
        )
        if config.gen_image_condition_refiner_config is not None:
            self.gen_image_condition_refiner = SimpleQFormerImageRefiner(
                hidden_size=self.llm_model.config.hidden_size, **config.gen_image_condition_refiner_config
            )

        # Regular PreTrainedModel post init, without which the weight init and tp_plan init will not be executed.
        self.post_init()

        # Runtime update the _no_split_modules for FSDP wrapping since Mammothmoda2Model has dynamic sub-classes.
        self._no_split_modules = list(
            dict.fromkeys(  # remove duplicates via dict.fromkeys()
                self._no_split_modules
                + getattr(self.llm_model, "_no_split_modules", [])
                + getattr(self.gen_transformer, "_no_split_modules", [])
            )
        )

    @property
    def input_embeddings(self) -> torch.Tensor:
        """LLM input embeddings property."""
        return self.get_input_embeddings()

    def get_output_embeddings(self) -> torch.Tensor:
        """Get LLM output embeddings."""
        return self.llm_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings) -> None:
        """Set LLM output embeddings."""
        self.llm_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder) -> None:
        """Set LLM decoder."""
        self.llm_model.set_decoder(decoder)

    def get_decoder(self) -> nn.Module:
        """Get LLM decoder."""
        return self.llm_model.get_decoder()

    def encode_vae(self, img: torch.Tensor) -> torch.Tensor:
        """Encode images using VAE."""
        if self.gen_vae.device != img.device:
            self.gen_vae = self.gen_vae.to(img.device)
        with torch.inference_mode():
            vae_output = self.gen_vae.encode(img)
            # Handle different VAE output formats
            if hasattr(vae_output, "latent_dist"):
                z0 = vae_output.latent_dist.sample()
            else:
                error_msg = f"Unknown VAE output format: {type(vae_output)}"
                raise ValueError(error_msg)
            # Apply shift and scaling factors if they exist
            z0 = self._apply_vae_scaling(z0)
            return z0

    def _apply_vae_scaling(self, z0: torch.Tensor) -> torch.Tensor:
        """Apply VAE shift and scaling factors."""
        config = getattr(self.gen_vae, "config", {})
        if isinstance(config, dict):
            if "shift_factor" in config and config["shift_factor"] is not None:
                z0 = z0 - config["shift_factor"]
            if "scaling_factor" in config and config["scaling_factor"] is not None:
                z0 = z0 * config["scaling_factor"]
        return z0

    def reinit_caption_embedder(self, in_features: int) -> None:
        """Reinitialize caption_embedder to adapt to Qwen2.5-VL."""
        logger.info("Reinitializing caption_embedder")
        out_features = self.gen_transformer.hidden_size
        self.gen_transformer.time_caption_embed.caption_embedder = nn.Sequential(
            Qwen2RMSNorm(in_features, eps=1e-05),
            nn.Linear(in_features, out_features, bias=True),
        )
        # Initialize caption_embedder weights
        nn.init.trunc_normal_(self.gen_transformer.time_caption_embed.caption_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.gen_transformer.time_caption_embed.caption_embedder[1].bias)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> "tuple | Mammothmoda2Qwen2_5_VLCausalLMOutputWithPast":
        """Part of Mammothmoda2 model forward, which is actually useless for pure generate."""
        return self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=None,  # Infer only.
            return_dict=True,
            **kwargs,
        )

    @torch.inference_mode()
    def generate(self, **kwargs) -> "GenerateOutput | torch.LongTensor":
        if kwargs.pop("t2i_generate", self.t2i_generate) is True:
            return self.generate_t2i(generation_config=self.generation_config, **kwargs)
        return self.llm_model.generate(generation_config=self.generation_config, **kwargs)

    @torch.inference_mode()
    def generate_t2i(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        ar_height: int = 32,
        ar_width: int = 32,
        cfg_scale: float = 6.0,
        **kwargs,
    ) -> None:
        generation_config.update(**kwargs)
        scope_start = generation_config.visual_token_start_id
        scope_end = scope_start + generation_config.visual_token_end_id
        B, L = input_ids.shape
        cfg_enable = cfg_scale > 1.0

        logits_processor = LogitsProcessorList()
        if cfg_enable:
            # NOTE: cfg_scale > 1.0, we need to mask the cfg attention mask
            assert B % 2 == 0, "Batch size must be even when using classifier free guidance"
            unguided_bsz = B // 2
            attention_mask[unguided_bsz:, :-10] = 0
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(guidance_scale=cfg_scale))
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            logits_processor.append(TemperatureLogitsWarper(temperature=generation_config.temperature))
        # NOTE: Must add before LogitsProcessor which filter the logits by relative rank
        logits_processor.append(SampledScopeLogitsProcessor(scope_start=scope_start, scope_end=scope_end))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            # min_tokens_to_keep = len(generation_config._eos_token_tensor) + 1
            logits_processor.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=3))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            logits_processor.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=3))

        step, next_tokens = 0, None
        generated_ids = input_ids.clone()
        for h in range(ar_height):
            for w in range(ar_width + 1):
                if step == 0:
                    cache_position = torch.arange(0, input_ids.shape[1], device=input_ids.device)
                    past_key_values = DynamicCache()
                    multimodal_extra_inputs = kwargs
                else:
                    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                    cache_position = torch.arange(past_seen_tokens, past_seen_tokens + 1, device=input_ids.device)
                    multimodal_extra_inputs = {}
                next_results = self.llm_model(
                    input_ids=input_ids if step == 0 else next_tokens,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    **multimodal_extra_inputs,
                )

                past_key_values = next_results.past_key_values
                logits = next_results.logits[:, -1, :].float()
                next_token_scores = logits_processor(generated_ids, logits)
                if generation_config.do_sample:  # sample
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:  # argmax
                    next_tokens = torch.argmax(next_token_scores, dim=-1, keepdim=True)

                if cfg_enable:
                    next_tokens[unguided_bsz:] = next_tokens[:unguided_bsz]

                # Assure eol_token in the end of each row
                if w == ar_width:
                    next_tokens[:] = generation_config.eol_token_id

                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=1)
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)
                step += 1
        return generated_ids, attention_mask
