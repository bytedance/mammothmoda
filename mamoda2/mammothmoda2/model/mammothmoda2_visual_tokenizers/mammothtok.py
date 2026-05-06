"""
MammothTok Model
"""

from dataclasses import dataclass, field
from typing import List, Optional

# from skimage.metrics import peak_signal_noise_ratio as psnr_loss
# from skimage.metrics import structural_similarity as ssim_loss
import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file
from torch import Tensor, nn

from .aimv2_models import aimv2_large_native

# TF32 config (for CUDA backends; might be ignored on NPU)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check flash-attention support
HAS_FLASH_ATTENTION_V2 = torch.__version__ >= "2.2.0"


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Depth-to-Space DCR mode (depth-column-row) core implementation.

    Args:
        x (torch.Tensor): input tensor. The channels-first (*CHW) layout is supported.
        block_size (int): block side size
    """
    # check inputs
    if x.dim() < 3:
        raise ValueError(f"Expecting a channels-first (*CHW) tensor of at least 3 dimensions")
    c, h, w = x.shape[-3:]

    s = block_size**2
    if c % s != 0:
        raise ValueError(f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels")

    outer_dims = x.shape[:-3]

    # splitting two additional dimensions from the channel dimension
    x = x.view(-1, block_size, block_size, c // s, h, w)

    # putting the two new dimensions along H and W
    x = x.permute(0, 3, 4, 1, 5, 2)

    # merging the two new dimensions with H and W
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)

    return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """
    modified from llamagen and magvit
    Args:
        affinity: (b, n, n), the affinity matrix, where affinity[i, j] is the affinity
                between encoed vector i and codebook vector j
        loss_type: how to turn the affinity into probability distribution
    """
    # shape: (b n) n
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    # target_probs.shape: (b, n, n), and sum(target_probs, dim=-1) = 1
    avg_probs = torch.mean(target_probs, dim=0)  # (,n)
    # average entropy corresponeds (negatively) to the diversity of indices for a single position
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    # sample entropy is the confidence for the quantization process
    # (bn, n) -> (bn) -> avg
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=1):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=0):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)  # (grid_size**2, head_dim // 2, 2)
    if cls_token_num > 0:
        cond_cache = torch.cat(
            [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
        )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
        return cond_cache
    else:
        return cache


def precompute_freqs_cis_2d_non_square(
    grid_size_y: int, grid_size_x: int, n_elem: int, base: int = 10000, cls_token_num=0
):
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))  # shape: [half_dim//2]

    # Y (height) and X (width) direction position indices
    y = torch.arange(grid_size_y)
    x = torch.arange(grid_size_x)

    # Expand to full grid
    freq_y = torch.outer(y, freqs)  # (grid_size_y, half_dim//2)
    freq_x = torch.outer(x, freqs)  # (grid_size_x, half_dim//2)

    # Broadcast and concat: each position has [freq_y, freq_x]
    freqs_grid = torch.cat(
        [
            freq_y[:, None, :].expand(-1, grid_size_x, -1),  # (grid_size_y, grid_size_x, half_dim//2)
            freq_x[None, :, :].expand(grid_size_y, -1, -1),  # (grid_size_y, grid_size_x, half_dim//2)
        ],
        dim=-1,
    )  # shape: [H, W, head_dim//2]

    # Final rotary embedding [cos, sin]
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1)  # shape: [H, W, head_dim//2, 2]
    cache = cache_grid.flatten(0, 1)  # shape: [H*W, head_dim//2, 2]

    if cls_token_num > 0:
        cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])
        return cond_cache
    else:
        return cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, bs_first=True):
    # if bs_first
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    # else
    # x: (seq_len, bs, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)  # (bs, seq_len, n_head, head_dim//2, 2)
    if bs_first:
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)
    else:
        freqs_cis = freqs_cis.view(xshaped.size(1), 1, 1, xshaped.size(3), 2)  # (1, seq_len, 1, head_dim//2, 2)

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)


class FractionalUpsample(nn.Module):
    """插值一个 fractional scale: 例如 7/8×"""

    def __init__(self, scale: float, mode="bilinear", align_corners=False):
        super().__init__()
        self.scale = scale
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        # fractional scale 插值
        return F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=self.align_corners)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class D2SUpsampler(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        """
        input_image: [B C H W]
        """
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        # self.lin = nn.Linear(z_channels, in_filters * 2)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps

    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        # quantizer = F.adaptive_avg_pool2d(quantizer, (1, 1))
        ### calculate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps  # not unbias
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        ### calculate mean for bias
        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)

        x = self.gn(x)
        x = scale * x + bias

        return x


class AttentionCustom(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        resid_dropout_p,
        use_rope=False,
        use_qk_norm=False,
        use_flash_attn=False,
        no_bias=False,
        attn_dropout_p=0,
    ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        Currently, the dimension of the key and value is the same as the query.
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.use_flash_attn:
            print("Using flash attention!")
        # flash attention can be switched to normal attention for inference
        # raise error only when training and use_flash_attn is True and
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not self.use_flash_attn) or (not self.training), (
            "Flash attention is not installed and cannot be used when training"
        )
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim = 3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.q_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.k_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.v_proj = nn.Linear(dim, dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        """
        The q, k, v will be projected into multiple heads.
        """
        seqlen, bsz, _ = query.shape

        # rearrange, (L, B, D) -> (B, L, D)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        xq = self.q_proj(query)
        xk = self.k_proj(key)
        xv = self.v_proj(value)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis)
            xk = apply_rotary_emb(xk, freqs_cis)
        else:
            assert freqs_cis is None, (
                "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
            )

        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=attn_mask,
                    is_causal=is_causal,  # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0,
                )
        else:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                is_causal=is_causal,  # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0,
            )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)


class SelfAttentionCustom(nn.Module):
    def __init__(
        self,
        dim,
        n_head,
        resid_dropout_p,
        use_rope=False,
        use_qk_norm=False,
        no_bias=False,
        use_flash_attn=False,
        attn_dropout_p=0,
    ):
        """
        This custom attention block supports the following modifications:
        - ROPE
        - QK Norm
        - Flash attention
        """
        super().__init__()

        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        # flash attention can be switched to normal attention for inference
        # raise error only when training and use_flash_attn is True and
        # flash attention is not installed
        assert HAS_FLASH_ATTENTION_V2 or (not use_flash_attn) or (not self.training), (
            "Flash attention is not installed and cannot be used when training"
        )
        assert dim % n_head == 0

        if self.use_flash_attn:
            print("Using flash attention!")

        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head
        total_qkv_dim = 3 * self.n_head * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(dim, total_qkv_dim, bias=not no_bias)
        self.wo = nn.Linear(dim, dim, bias=not no_bias)

        # regularization
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        seqlen, bsz, _ = x.shape

        # rearrange, (L, B, D) -> (B, L, D)
        x = x.transpose(0, 1)

        xq, xk, xv = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim)

        if self.use_qk_norm:
            xq = self.q_norm(xq)
            xk = self.k_norm(xk)

        if self.use_rope:
            xq = apply_rotary_emb(xq, freqs_cis, True)
            xk = apply_rotary_emb(xk, freqs_cis, True)
        else:
            assert freqs_cis is None, (
                "Attention Module is not using ROPE but freqs_cis is not None. Check your setting!"
            )

        # (B, L, H, D) -> (B, H, L, D)
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.use_flash_attn:
            # Shape: (batch_size, num_heads, seq_length, head_dim)
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
                output = F.scaled_dot_product_attention(
                    xq,
                    xk,
                    xv,
                    attn_mask=mask,
                    is_causal=is_causal,  # is_causal=False is for KV cache
                    dropout_p=self.attn_dropout_p if self.training else 0,
                )
        else:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=mask,
                is_causal=is_causal,  # is_causal=False is for KV cache
                dropout_p=self.attn_dropout_p if self.training else 0,
            )

        # (B, H, L, D) -> (B, L, H*D)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))

        # rearrange back, (B, L, D) -> (L, B, D)
        return output.transpose(0, 1)


class TransformerDecoderLayer(nn.Module):
    """
    This is the Q-former layer from DETR.
    """

    def __init__(
        self,
        d_model,
        nhead,
        mlp_ratio=4.0,
        dropout=0.1,
        activation=nn.GELU,
        normalize_before=True,
        query_rope=False,
        use_qk_norm=False,
        use_flash_attn=False,
    ):
        super().__init__()
        self.query_rope = query_rope
        self.use_qk_norm = use_qk_norm
        self.use_flash_attn = use_flash_attn
        if self.query_rope:
            self.self_attn = SelfAttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_rope=True,
            )
            self.multihead_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_rope=True,
            )
        elif self.use_qk_norm or self.use_flash_attn:
            self.self_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_qk_norm=self.use_qk_norm,
                use_flash_attn=self.use_flash_attn,
            )
            self.multihead_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_qk_norm=self.use_qk_norm,
                use_flash_attn=self.use_flash_attn,
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False, bs_first=True):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=bs_first)
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.query_rope:
            tgt2 = self.self_attn(
                tgt,
                freqs_cis=query_pos,
                mask=tgt_mask,
            )
        elif self.use_qk_norm or self.use_flash_attn:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.query_rope:
            tgt2 = self.multihead_attn(
                query=tgt,
                key=memory,
                value=memory,
                attn_mask=memory_mask,
                freqs_cis=pos,
            )
        elif self.use_qk_norm or self.use_flash_attn:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos, rope=self.query_rope),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
            )
        else:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        if self.query_rope:
            tgt2 = self.self_attn(tgt2, query_pos, tgt_mask)
        elif self.use_qk_norm or self.use_flash_attn:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.query_rope:
            tgt2 = self.multihead_attn(
                query=tgt2,
                key=memory,
                value=memory,
                attn_mask=memory_mask,
                freqs_cis=pos,
            )
        elif self.use_qk_norm or self.use_flash_attn:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos, rope=self.query_rope),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
            )
        else:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, query_pos),
                key=self.with_pos_embed(memory, pos),
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
        )


class TransformerLayer(nn.Module):
    """
    This is the standard Transformer layer. Currently only supports absolute positional embedding.
    # TODO: support RoPE 2d
    """

    def __init__(
        self,
        d_model,
        nhead,
        mlp_ratio=4.0,
        dropout=0.1,
        activation=nn.GELU,
        normalize_before=True,
        query_rope=False,
        use_qk_norm=False,
    ):
        super().__init__()
        self.use_rope = query_rope
        # assert not self.use_rope, "QformerLayerVar does not support RoPE yet"
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.self_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_qk_norm=self.use_qk_norm,
                use_flash_attn=self.use_flash_attn,
            )
        elif self.use_rope:
            self.self_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_rope=True,
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.use_qk_norm:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        elif self.use_rope:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, freqs_cis=pos)
        else:
            q = k = tgt
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        if self.use_qk_norm:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask)
        elif self.use_rope:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask, freqs_cis=pos)
        else:
            tgt2 = self.self_attn(tgt2, tgt2, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # assert pos is None, "TransformerLayer does not support injected positional embedding"
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, pos)


class QformerLayerVar(nn.Module):
    """
    This is the variant of Q-former layer from DETR. It is merely used for comparison.
    This layer has 2 multi-head attention layers, and the positional embedding follows the original DETR.
    But there is not reference feature map any more. All attentions are self-attention.
    """

    def __init__(
        self,
        d_model,
        nhead,
        mlp_ratio=4.0,
        dropout=0.1,
        activation=nn.GELU,
        normalize_before=True,
        use_rope=False,
        use_qk_norm=False,
    ):
        super().__init__()
        self.use_rope = use_rope
        assert not self.use_rope, "QformerLayerVar does not support RoPE yet"
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.self_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_qk_norm=self.use_qk_norm,
                use_flash_attn=self.use_flash_attn,
            )
            self.multihead_attn = AttentionCustom(
                d_model,
                nhead,
                resid_dropout_p=dropout,
                use_qk_norm=self.use_qk_norm,
                use_flash_attn=self.use_flash_attn,
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        dim_feedforward = int(d_model * mlp_ratio)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor], rope=False):
        if rope:
            return apply_rotary_emb(tensor, pos, bs_first=False)
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.use_qk_norm:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if self.use_qk_norm:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, pos, rope=self.use_rope),
                key=self.with_pos_embed(tgt, pos),
                value=tgt,
                attn_mask=tgt_mask,
            )
        else:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt, pos, rope=self.use_rope),
                key=self.with_pos_embed(tgt, pos),
                value=tgt,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        if self.use_qk_norm:
            q = k = self.with_pos_embed(tgt, pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask)
        else:
            q = k = self.with_pos_embed(tgt2, pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        if self.use_qk_norm:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, pos, rope=self.use_rope),
                key=self.with_pos_embed(tgt2, pos),
                value=tgt2,
                attn_mask=tgt_mask,
            )
        else:
            tgt2 = self.multihead_attn(
                query=self.with_pos_embed(tgt2, pos, rope=self.use_rope),
                key=self.with_pos_embed(tgt2, pos),
                value=tgt2,
                attn_mask=tgt_mask,
                key_padding_mask=tgt_key_padding_mask,
            )[0]

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, pos)


class ViTDecoder2D(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        model_size="small",
        token_size=256,
        dropout=0.0,
        out_inner_feat=False,
        out_inner_dim=768,  # for dino-v2
        out_inner_depth=None,
        transformer_layer_type="QformerLayerVar",
        use_rope=False,
        rope_1d=False,
    ):
        super().__init__()
        self.transformer_layer_type = transformer_layer_type
        assert self.transformer_layer_type in ["QformerLayerVar", "TransformerLayer", "TransformerDecoderLayer"]

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = model_size
        self.token_size = token_size
        self.out_inner_feat = out_inner_feat
        self.out_inner_depth = out_inner_depth
        self.use_rope = use_rope
        self.rope_1d = rope_1d
        self.width = {"tiny": 256, "small": 512, "base": 768, "large": 1024, "xl": 1280, "xxl": 1536, "xxxl": 2560}[
            self.model_size
        ]
        self.num_layers = {"tiny": 4, "small": 6, "base": 12, "large": 24, "xl": 36, "xxl": 48, "xxxl": 48}[
            self.model_size
        ]
        self.num_heads = {"tiny": 4, "small": 8, "base": 12, "large": 16, "xl": 20, "xxl": 24, "xxxl": 40}[
            self.model_size
        ]

        self.decoder_embed = nn.Linear(self.token_size, self.width, bias=True)
        self.ln_pre = nn.LayerNorm(self.token_size)
        scale = self.width**-0.5

        if self.use_rope:
            if rope_1d:
                pass
                # print("Using 1D RoPE")
                # self.freqs_cis = precompute_freqs_cis(self.grid_size**2,
                #                                       self.width // self.num_heads,
                #                                       base=10000, cls_token_num=0)
            else:
                pass
                # print("Using 2D RoPE")
                # self.freqs_cis = precompute_freqs_cis_2d(self.grid_size,
                #                                             self.width // self.num_heads,
                #                                             base=10000,
                #                                             cls_token_num=0)
        else:
            self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size**2, 1, self.width))
        self.transformer = nn.ModuleList()
        layer_cls = eval(self.transformer_layer_type)
        for i in range(self.num_layers):
            self.transformer.append(
                layer_cls(
                    self.width,
                    self.num_heads,
                    mlp_ratio=4.0,
                    dropout=dropout,
                    query_rope=self.use_rope,
                )
            )
        self.ln_post = nn.LayerNorm(self.width)

        self.conv_out = nn.Conv2d(self.width, token_size, kernel_size=3, stride=1, padding=1)

        if out_inner_feat:
            self.distill_mlp = nn.Sequential(
                nn.Linear(self.width, self.width * 4),
                nn.SiLU(),
                nn.Linear(self.width * 4, self.width * 4),
                nn.SiLU(),
                nn.Linear(self.width * 4, out_inner_dim),
            )

    def forward(self, z_quantized, ret_inner_feat=False):
        N, C, H, W = z_quantized.shape
        selected_latent_tokens = W
        x = z_quantized.reshape(N, C, H * W).permute(2, 0, 1)  # LND
        x = self.decoder_embed(self.ln_pre(x))

        seq_len, bs, _ = x.shape  # shape: (num_latent_tokens, B, c)
        latent_tokens = x

        if self.use_rope:
            # pos_embed = self.freqs_cis.to(x.dtype).to(x.device)
            if self.rope_1d:
                pos_embed = (
                    precompute_freqs_cis(H * W, self.width // self.num_heads, base=10000, cls_token_num=0)
                    .to(x.dtype)
                    .to(x.device)
                )
            else:
                if H != W:
                    pos_embed = (
                        precompute_freqs_cis_2d_non_square(
                            H, W, self.width // self.num_heads, base=10000, cls_token_num=0
                        )
                        .to(x.dtype)
                        .to(x.device)
                    )
                else:
                    pos_embed = (
                        precompute_freqs_cis_2d(H, self.width // self.num_heads, base=10000, cls_token_num=0)
                        .to(x.dtype)
                        .to(x.device)
                    )
        else:
            pos_embed = self.positional_embedding.repeat(1, bs, 1).to(x.dtype)  # shape = [*, grid ** 2 + 1, width]
        # if self.transformer_layer_type == "TransformerLayer":
        #     # this is the original transformer layer, use absolute position embedding
        #     latent_tokens = latent_tokens + pos_embed

        for i in range(self.num_layers):
            if self.transformer_layer_type == "TransformerDecoderLayer":
                latent_tokens = self.transformer[i](
                    latent_tokens, latent_tokens, pos=pos_embed, query_pos=pos_embed, tgt_mask=None
                )
            elif self.transformer_layer_type == "TransformerLayer":
                latent_tokens = self.transformer[i](latent_tokens, pos=pos_embed)
            elif self.transformer_layer_type == "QformerLayerVar":
                latent_tokens = self.transformer[i](latent_tokens, pos=pos_embed)
            if self.out_inner_feat and ret_inner_feat and (i + 1) == self.out_inner_depth:
                inner_feat = self.distill_mlp(latent_tokens)

        latent_tokens = self.ln_post(latent_tokens)
        # L N D -> N D H W
        # latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, self.grid_size, self.grid_size)
        latent_tokens = latent_tokens.permute(1, 2, 0).reshape(bs, self.width, H, W)
        latent_tokens = self.conv_out(latent_tokens.contiguous())
        if self.out_inner_feat and ret_inner_feat:
            # L N D -> N L D
            inner_feat = inner_feat.permute(1, 0, 2)
            return latent_tokens, inner_feat
        else:
            return latent_tokens


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
        adaptive_gn=False,
        d2s_up=False,
        use_attn=True,
        res_up_sample=False,
        upsample_match_channel=False,
        is_aimv2=False,
    ):
        """
        adaptive_gn: whether to use adaptive group normalization as in MAGVIT-v2
        d2s_up: whether to use depth_to_space for up sampling
        res_up: whether to use residual non-parametric depth-to-space when upsampling
        """
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.is_aimv2 = is_aimv2

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.adaptive_gn = adaptive_gn
        self.d2s_up = d2s_up

        self.use_attn = use_attn

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        if self.use_attn:
            self.mid.append(AttnBlock(block_in, norm_type=norm_type))

        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))

        # upsampling
        prev_out_dim = block_in

        self.conv_blocks = nn.ModuleList()
        if self.adaptive_gn:
            self.adaptive = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            if self.use_attn:
                attn_block = nn.ModuleList()
            block_in = prev_out_dim
            block_out = ch * ch_mult[i_level]
            if self.adaptive_gn:
                self.adaptive.append(AdaptiveGroupNorm(z_channels, block_in))
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if self.use_attn and i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            if self.use_attn:
                conv_block.attn = attn_block
            # upsample
            if i_level != 0:
                if d2s_up:
                    assert not res_up_sample, "d2s_up or res_up_sample can not be True at the same time"
                    conv_block.upsample = D2SUpsampler(block_in)
                    prev_out_dim = block_in
                elif res_up_sample:
                    if upsample_match_channel:
                        conv_block.upsample = UpsamplerWithPixshuffleDupResidual(
                            block_in,
                            ch * ch_mult[i_level - 1],
                        )
                        prev_out_dim = ch * ch_mult[i_level - 1]
                    else:
                        conv_block.upsample = UpsamplerWithPixshuffleDupResidual(block_in)
                        prev_out_dim = block_in
                else:
                    conv_block.upsample = Upsample(block_in, resamp_with_conv)
                    prev_out_dim = block_in

            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        if self.adaptive_gn:
            style = z.clone()

        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            if self.adaptive_gn:
                ### pass in each resblock first adaGN
                try:
                    h = self.adaptive[i_level](h, style)
                except Exception as e:
                    error_info = str(e) + f"Showing the h shape: {h.shape}, {style.shape}"
                    raise ValueError(error_info)
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if self.use_attn and len(block.attn) > 0:
                    h = block.attn[i_block](h)
            # AIMv2 16上采样率-> 14上采样率
            if self.is_aimv2 and i_level == self.num_resolutions - 2:
                h = FractionalUpsample(7 / 8)(h)

            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type="group"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class UpsamplerWithPixshuffleDupResidual(nn.Module):
    """
    Using pixshuffle for upsampling, and with a residual connection.
    The residual is the depth to space + channel duplicated value
    """

    def __init__(
        self,
        dim,
        dim_out=None,
        factor=2,  # the spatial upsampling factor
    ):
        super().__init__()

        # for the main upsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.factor = factor
        self.conv1 = nn.Conv2d(dim, self.dim_out * factor**2, (3, 3), padding=1)

        # for residual non-parameteric connections
        # note we
        assert self.dim_out * factor**2 % dim == 0
        self.repeats = self.dim_out * factor**2 // dim

    def forward(self, x: torch.Tensor):
        """
        input_image: [B C H W]
        """
        # we use the implementation from efficientvit/models/nn/ops: first duplicate then shuffle
        # but this is not exactly the same as the LARP paper presents(first shuffle then duplicate).
        residual = x.repeat_interleave(self.repeats, dim=1)
        residual = F.pixel_shuffle(residual, self.factor)

        out = self.conv1(x)
        out = F.pixel_shuffle(out, self.factor)
        return out + residual


class DownsamplerWithPixunshuffleResidual(nn.Module):
    """
    Using pixshuffle for upsampling, and with a residual connection.
    The residual is the depth to space + channel duplicated value
    """

    def __init__(
        self,
        dim,
        dim_out=None,
        factor=2,  # the spatial downsampling factor
    ):
        super().__init__()

        # for the main downsampler
        self.dim_out = dim if dim_out is None else dim_out
        self.factor = factor
        self.conv1 = nn.Conv2d(dim, self.dim_out // factor**2, (3, 3), padding=1)

        # for residual non-parameteric connections
        # note we
        self.factor = factor
        assert dim * factor**2 % self.dim_out == 0
        self.group_size = dim * factor**2 // self.dim_out

    def forward(self, x: torch.Tensor):
        """
        input_image: [B C H W]
        """
        residual = F.pixel_unshuffle(x, self.factor)
        B, C, H, W = residual.shape
        residual = residual.view(B, self.dim_out, self.group_size, H, W)
        residual = residual.mean(dim=2)

        out = self.conv1(x)
        out = F.pixel_unshuffle(out, self.factor)
        return out + residual


class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class QFormerUpsample(nn.Module):
    """
    Cross attn module as an Upsampler
    """

    def __init__(
        self,
        width,
        nhead,
        dropout,
        num_queries=256,
        mlp_ratio=4.0,
        activation=nn.GELU,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(width, nhead, dropout=dropout)
        scale = width**-0.5
        self.num_queries = num_queries
        self.query_1d = nn.Parameter(scale * torch.randn(num_queries, 1, width))

        self.pos_emb = nn.Parameter(scale * torch.randn(num_queries, 1, width))

        dim_feedforward = int(width * mlp_ratio)
        self.linear1 = nn.Linear(width, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, width)

        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        # self.dropout1 = nn.Dropout(dropout)

        self.activation = activation()

    def forward(self, x):
        if x.shape[0] == self.num_queries:
            return x
        else:
            x_2 = self.norm1(x)
            x_2 = self.pos_emb[: x.shape[0]].repeat(1, x.shape[1], 1).to(x.dtype) + x_2
            query = self.query_1d[x.shape[0] :].repeat(1, x.shape[1], 1).to(x.dtype)
            x_2 = self.cross_attn(query=query, key=x_2, value=x_2)[0]
            x_2 = self.norm2(x_2)
            x_2 = self.linear2(self.dropout(self.activation(self.linear1(x_2))))
            # concat at the first dimension
            x = torch.cat([x, x_2], dim=0)
            return x


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        entropy_loss_ratio,
        l2_norm,
        show_usage,
        rot=False,
        stochastic=False,
        stochastic_temperature=1.0,
        eval_deterministic=False,
        simvq=False,
        codebook_transform=None,
        freeze_codebook=False,
    ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
        """

        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0:  # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else:  # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = (
            e
            - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1))
            + 2 * torch.bmm(torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        )
        return e

    def forward(self, z, random_replace=False, replace_ratio=0.1):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)

            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(embedding**2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding))
            )

            min_encoding_indices = torch.argmin(d, dim=1)  # (b*h*w)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        b, h, w, c = z.shape
        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, "b h w c-> (b h w) c")
            z_q = rearrange(z_q, "b h w c -> (b h w) c")
            pre_norm_q = self.get_very_efficient_rotation(
                z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                z.unsqueeze(1),
            ).squeeze()
            z_q = (
                pre_norm_q
                * (torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            )
            z_q = rearrange(z_q, "(b h w) c -> b h w c", b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()

        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, "(b h w) c -> b h w c", b=b, h=h, w=w)
            mask = (
                torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)
            )  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)

        # reshape back to match original input shape
        z_q = torch.einsum("b h w c -> b c h w", z_q)

        return (
            z_q,
            [vq_loss, commit_loss, entropy_loss, codebook_usage],
            (perplexity, min_encodings, min_encoding_indices),
        )

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


@dataclass
class VQVitModelPlusArgs:
    # for quantization
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    # tricks for lfq. deprecated
    use_lfq: bool = False
    bernoulli_sample: bool = False
    eval_deterministic: bool = False

    # SimVQ trick, deprecated
    simvq: bool = False
    codebook_transform: str = None
    freeze_codebook: bool = False

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = "small"
    encoder_size: str = None
    decoder_size: str = None
    num_latent_tokens: int = 256
    z_channels: int = 256  # the dimension of the intermediate downsample towards codebook dimension
    dropout_p: float = 0.0

    # use rope for the decoder Q-former attention.
    use_rope: bool = False
    use_qk_norm: bool = False

    # TODO: remove option. flash attention is automatically used when calling scaled_dot_product_attention
    use_flash_attn: bool = False

    # the setting for initializing the 1d queries for the 2dto1d encoder
    multi_level_query_init: bool = False
    learnable_1d_query_init: bool = False
    rope_1d: bool = False

    # the initialization for the 2d queries. the "level" corresponds to the
    # "level" division for "multi_level_query_init". It assumes 1d tokens have levels
    # all false means simply using the global average of the 1d tokens to initialize
    # the 2d queries.
    last_level_2d_query_init: bool = False
    multi_level_2d_query_init: bool = False
    learnable_2d_query_init: bool = False

    # tricks for the CNN 2d decoder
    adaptive_gn: bool = False
    d2s_up: bool = False
    res_up_down_sample: bool = False
    downsample_match_channel: bool = False
    upsample_match_channel: bool = False
    res_codebook_updown_sample: bool = False
    downsample_improve: bool = False
    # whether to use attention in the 2d encoder or decoder
    # suggested not to. May be unstable and slower
    use_attn: bool = True

    # rope 2d only supports the 1dto2d decoder queries (since Q-former)
    rope_2d: bool = False

    # the rotation trick for quantizer. The influence is limited
    rot: bool = False

    # for stochastic quantization. Closed by default
    stochastic: bool = False
    stochastic_temperature: float = 0.03

    # distillation setting
    distill_depth: int = None
    # whether to distill from encoder. Not tested yet.
    # (to be deleted)
    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0

    # for gptc model, which tries to utilize AR prior for
    # training tokenizers. The effect is limited and this feature
    # is deprecated.
    # for ar prior model
    with_prior_model: bool = False
    prior_model_config: dict = None

    image_size: int = 256


@dataclass
class VQVitModel2DPlusArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = False
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = "small"
    num_latent_tokens: int = 256
    encoder_size: str = None
    decoder_size: str = None
    transformer_layer_type: str = "TransformerDecoderLayer"
    z_channels: int = 256
    dropout_p: float = 0.0

    adaptive_gn: bool = False
    d2s_up: bool = False

    rot: bool = False
    distill_depth: int = None

    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0
    use_attn: bool = True

    use_rope: bool = False
    image_size: int = 256
    rope_1d: bool = False
    use_flash_attn: bool = False
    pretrain_encoder_path: str = None
    use_pretrain_encoder: bool = False
    up_sample_ratio: int = 16


class VQVitModel2DPlus_AIMv2(nn.Module):
    def __init__(self, config: VQVitModel2DPlusArgs):
        super().__init__()
        self.config = config
        logger.info(f"AIMv2 Tokenizer config: {config}")

        self.encoder = aimv2_large_native()
        self.is_native = True
        decoder_size = config.decoder_size if config.decoder_size is not None else config.model_size

        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None

        if config.encoder_2d_distill:
            assert config.distill_depth is None

        if config.encoder_2d_distill:
            self.distill_mlp = nn.Sequential(
                nn.Linear(config.z_channels, config.z_channels * 4),
                nn.SiLU(),
                nn.Linear(config.z_channels * 4, config.z_channels * 4),
                nn.SiLU(),
                nn.Linear(config.z_channels * 4, config.out_inner_dim),
            )

        self.s2ddecoder = ViTDecoder2D(
            image_size=config.image_size,
            model_size=decoder_size,
            token_size=config.z_channels,
            dropout=config.dropout_p,
            patch_size=16,
            out_inner_feat=config.distill_depth is not None,
            out_inner_depth=config.distill_depth,
            out_inner_dim=config.out_inner_dim,
            transformer_layer_type=config.transformer_layer_type,
            use_rope=config.use_rope,
            rope_1d=config.rope_1d,
        )

        self.decoder = Decoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
            adaptive_gn=config.adaptive_gn,
            d2s_up=config.d2s_up,
            use_attn=config.use_attn,
            is_aimv2=False if self.is_native and config.up_sample_ratio == 16 else True,
        )

        # self.apply(self._init_weights)

        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
            rot=config.rot,
        )
        self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        # for name, module in self.named_modules():
        #     module.register_forward_hook(nan_hook)
        if config.use_pretrain_encoder:
            self.load_pretrain_encoder(config.pretrain_encoder_path)

        # print("encoder:\n", self.encoder)

    def load_pretrain_encoder(
        self,
        pretrain_encoder_path="/mnt/bn/seutao-hl/chentaicai/MammothModa-U/tokenversa/GigaTok/checkpoints/model.safetensors",
    ):
        if pretrain_encoder_path is None:
            return
        print("load pretrain encoder from: ", pretrain_encoder_path)
        state_dict = load_file(pretrain_encoder_path)
        self.encoder.load_state_dict(state_dict=state_dict, strict=True)

    def _init_weights(self, module):
        """Initialize the weights.
        :param:
            module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, return_code=True, return_feat=False, random_mix_reg=False, replace_ratio=0.1, **kwargs):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        B, C, H, W = x.shape
        Hd, Wd = int(H / 14), int(W / 14)
        if return_feat:
            h = self.encoder(x)
            # # 假设你已经知道 B, N, C
            # B, N, C = h.shape
            # # 假设你知道 H 和 W（如 H=W=int(sqrt(N))）
            # H = W = int(N ** 0.5)
            # assert H * W == N, "N 必须是 H×W 的平方数"
            # # 转换形状
            # h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # → [B, C, H, W]
            return h, None, None

        h = self.encoder(x)
        # 假设你已经知道 B, N, C
        B, N, C = h.shape
        # 假设你知道 H 和 W（如 H=W=int(sqrt(N))）
        # H = W = int(N ** 0.5)
        assert Hd * Wd == N, "N 必须是 H×W 的平方数"
        # 转换形状
        h = h.view(B, Hd, Wd, C).permute(0, 3, 1, 2)  # → [B, C, H, W]
        # import pdb; pdb.set_trace()
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info

        return quant, emb_loss

    def decode(self, quant, ret_inner_feat=False, return_feat=False):
        quant = self.post_quant_conv(quant)
        # import pdb; pdb.set_trace()
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s2ddecoder(quant, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe or visualization (don not go through mlp)
            _, feat = self.s2ddecoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return _, feat
        else:
            rec_spatial = self.s2ddecoder(quant)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
        self,
        input,
        num_en_q_level=None,
        causal_type=None,
        rec_loss=True,
        ret_inner_feat=False,
        random_mix_reg=False,
        replace_ratio=None,
        global_step=None,
        max_steps=None,
    ):
        quant, diff = self.encode(input, return_code=False, random_mix_reg=random_mix_reg, replace_ratio=replace_ratio)
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                dec, rec_spatial = self.decode(quant)
            else:
                dec, rec_spatial, inner_feat = self.decode(quant, ret_inner_feat=True)
        else:
            dec, rec_spatial = self.decode(quant)

        if self.training:
            fea_rec_loss = 0

        if self.training:
            dir_dec = None

            if ret_inner_feat:
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            return [dec, dir_dec], [diff, fea_rec_loss]

        return dec, diff