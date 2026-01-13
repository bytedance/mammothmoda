# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers.transformer_wan import  WanAttnProcessor, WanAttention, WanImageEmbedding, WanRotaryPosEmbed
from .moe_layer import MoELayer
from .config import MoELayerConfig, VLMTokenRefinerConfig
from .vlm_refiner import VLMTokenRefiner

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states_image


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # MoE Layer
        if num_heads == 12:
            moe_config = MoE_CONFIG_6B_A1B 
        else:
            moe_config = MoE_CONFIG_20B_A3B

        self.ffn = MoELayer(config=moe_config)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        self_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, self_attn_mask, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "blocks.*": {
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

        # 5. VLM token refiner
        connector_config = VLMTokenRefinerConfig(enable=True).to_dict()
        self.converter_mlp = VLMTokenRefiner(
                in_dim=connector_config.get("vlm_dim", 4096),
                out_dim=text_dim,
                connector_config=connector_config,
            )
        
        # In context
        self.text_len = 512
        self.in_context_proj = nn.Linear(text_dim, inner_dim)
        # Learnable position embedding for VLM tokens in in_context mode
        self.vlm_pos_embed = nn.Parameter(
            torch.randn(1, self.text_len, inner_dim) * 0.02
        )
        self.hidden_size = inner_dim

    def generate_mask_from_embedding(self, prompt_embeds):
        if not isinstance(prompt_embeds, torch.Tensor):
            raise TypeError(f"prompt_embeds必须是torch.Tensor，当前类型：{type(prompt_embeds)}")
        if len(prompt_embeds.shape) != 3:
            raise ValueError(f"prompt_embeds必须是3维张量 [batch, seq_len, hidden_dim]，当前shape：{prompt_embeds.shape}")
        
        mask = (prompt_embeds != 0).any(dim=-1).long()
        mask = mask.to(prompt_embeds.device)
        return mask
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if encoder_hidden_states.device != hidden_states.device:
            encoder_hidden_states = encoder_hidden_states.to(device=hidden_states.device)

        prompt_mask = self.generate_mask_from_embedding(encoder_hidden_states)
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        encoder_hidden_states = encoder_hidden_states.unsqueeze(0)
        self.converter_mlp = self.converter_mlp.to(device=hidden_states.device, dtype=torch.bfloat16)
        encoder_hidden_states = self.converter_mlp(encoder_hidden_states)
        vlm_context_emb = None
        vlm_seq_len = self.text_len

        # breakpoint()
        # prompt embeddings
        bs = encoder_hidden_states.size(0)
        encoder_hidden_states = encoder_hidden_states.view(bs, -1, encoder_hidden_states.size(-1))
        if prompt_mask is not None:
            seq_lens = prompt_mask.view(bs, -1).sum(dim=-1)
            seq_lens = seq_lens.to(torch.int64)
            for i, seq_len in enumerate(seq_lens):
                encoder_hidden_states[i, seq_len:] = 0
        # breakpoint()
        # encoder_hidden_states = encoder_hidden_states.squeeze(0)

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )

        # In-context mode: project VLM features to hidden_size and concat to patch embs
        vlm_context_emb = self.in_context_proj(encoder_hidden_states.to(timestep_proj.dtype))
        vlm_context_emb = vlm_context_emb + self.vlm_pos_embed[:, :vlm_context_emb.size(1), :]
        # Keep vlm_seq_len fixed for PP compatibility (truncate/clip to text_len)
        vlm_seq_len = min(vlm_context_emb.size(1), self.text_len)
        vlm_context_emb = vlm_context_emb[:, :vlm_seq_len, :].contiguous()
        # Keep pipeline interface stable: always return a prompt_emb tensor

        encoder_hidden_states = torch.zeros(
            (batch_size, self.text_len, self.hidden_size),
            device=timestep_proj.device,
            dtype=timestep_proj.dtype,
        )
        # Concat VLM tokens before video tokens: [vlm_tokens, video_tokens]
        hidden_states = torch.cat([vlm_context_emb, hidden_states], dim=1)

        #For in_context mode, we need to extend rotary_emb to cover VLM tokens
        # VLM tokens use zero/identity position embeddings (no rotation)

        # Create identity rotary embeddings for VLM tokens (no positional rotation).
        # NOTE: RoPE freqs are complex64 in this codebase; on NPU, `torch.ones(dtype=complex64)`
        # is not supported. Work around by creating float ones then casting to complex (1+0j).

        freqs_cos, freqs_sin = rotary_emb
        vlm_rotary_zeros_cos = torch.ones(
            (batch_size, vlm_seq_len, 1, freqs_cos.shape[-1]),
            device=freqs_cos.device, dtype=freqs_cos.dtype
        )
        vlm_rotary_zeros_sin = torch.zeros(
            (batch_size, vlm_seq_len, 1, freqs_sin.shape[-1]),
            device=freqs_sin.device, dtype=freqs_sin.dtype
        )
        freqs_cos = torch.cat([vlm_rotary_zeros_cos, freqs_cos], dim=1)
        freqs_sin = torch.cat([vlm_rotary_zeros_sin, freqs_sin], dim=1)
        rotary_emb = (freqs_cos, freqs_sin)

        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        # 
        if prompt_mask is not None:
            p_mask = prompt_mask.view(batch_size, -1)
            # truncate or pad p_mask to match vlm_seq_len
            if p_mask.size(1) > vlm_seq_len:
                p_mask = p_mask[:, :vlm_seq_len]
            elif p_mask.size(1) < vlm_seq_len:
                padding = torch.zeros((batch_size, vlm_seq_len - p_mask.size(1)), device=p_mask.device, dtype=p_mask.dtype)
                p_mask = torch.cat([p_mask, padding], dim=1)
            
            # Full sequence mask: [vlm_tokens, video_tokens]
            # prompt_mask: 1 is valid, 0 is padding.
            # atten_mask for NPU fusion attention: True/1 is masked out, False/0 is valid.
            vlm_padding_mask = (p_mask == 0)
            video_seq_len = hidden_states.shape[1] - vlm_seq_len
            video_padding_mask = torch.zeros((batch_size, video_seq_len), device=p_mask.device, dtype=torch.bool)
            full_padding_mask = torch.cat([vlm_padding_mask, video_padding_mask], dim=1) # [B, S]
            
            # Expand to [B, 1, S, S] for FlashAttention
            S = full_padding_mask.shape[-1]
            self_attn_mask = (~full_padding_mask).view(batch_size, 1, 1, S).expand(batch_size, 1, S, S).to(hidden_states.device)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attn_mask
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, self_attn_mask)

        hidden_states = hidden_states[:, vlm_seq_len:, :]

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


# 6B-A1B MoE Config
MoE_CONFIG_6B_A1B = MoELayerConfig(
    num_layers=32,
    hidden_size=1536,
    num_attention_heads=12,
    ffn_hidden_size=2048,
    num_moe_experts=32,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_grouped_gemm=False,
    moe_router_topk=4,
    moe_router_dtype="fp32",
    add_bias_linear=False,
    use_cpu_initialization=True,
    gated_linear_unit=False,
    params_dtype=torch.bfloat16,
    num_shared_experts=0
)

# 20B-A3B MoE Config
MoE_CONFIG_20B_A3B = MoELayerConfig(
    num_layers=30,
    hidden_size=3072,
    num_attention_heads=24,
    ffn_hidden_size=1920,
    num_moe_experts=48,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_grouped_gemm=False,
    moe_router_topk=5,
    moe_router_dtype="fp32",
    add_bias_linear=False,
    use_cpu_initialization=True,
    gated_linear_unit=False,
    params_dtype=torch.bfloat16,
    num_shared_experts=1,
    shared_expert_intermediate_size=2048,
)
