# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import functools
from typing import Any, Callable, Literal, Optional, Tuple, Union

from torch import nn

from .aimv2_layers import (
    Attention,
    AttentionPoolingClassifier,
    PatchEmbed,
    RMSNorm,
    SwiGLUFFN,
    Transformer,
    ViTPreprocessor,
)

__all__ = ["AIMv2VisionEncoder", "aimv2_large_native"]


ArrayLike = Any
Module = Callable[..., Any]


class AIMv2VisionMixin:
    preprocessor: Module
    trunk: Module
    head: Module

    def forward(
        self,
        input_pixels: ArrayLike,
        mask: Optional[ArrayLike] = None,
        output_features: bool = False,
    ) -> Union[ArrayLike, Tuple[ArrayLike, Tuple[ArrayLike, ...]]]:
        x = self.preprocessor(input_pixels)
        x, features = self.trunk(x, mask=mask)
        x = self.head(x)
        return (x, tuple(features)) if output_features else x


class AIMv2VisionEncoder(AIMv2VisionMixin, nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 14,
        embed_dim: int = 1024,
        mlp_hidden_dim: int = 2816,
        num_blocks: int = 24,
        num_heads: int = 8,
        num_channels: int = 3,
        head_num_heads: int = 8,
        head_num_queries: int = 1,
        head_average_pool: bool = True,
        head_linear_bias: bool = False,
        pos_embed_type: Literal["sincos", "absolute"] = "absolute",
        head_type: Optional[Literal["attention-pool"]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        norm_layer = functools.partial(RMSNorm, eps=1e-5)
        patchifier = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.preprocessor = ViTPreprocessor(
            patchifier,
            drop_patches=False,
            cls_token=False,
            pos_embed_type=pos_embed_type,
        )
        self.trunk = Transformer(
            attn_target=lambda use_bias: Attention(dim=embed_dim, num_heads=num_heads, use_bias=use_bias),
            ffn_target=SwiGLUFFN,
            embed_dim=embed_dim,
            mlp_hidden_dim=mlp_hidden_dim,
            num_blocks=num_blocks,
            norm_layer=norm_layer,
            **kwargs,
        )
        if head_type == "attention-pool":
            self.head = AttentionPoolingClassifier(
                embed_dim,
                out_features=embed_dim,
                num_heads=head_num_heads,
                num_queries=head_num_queries,
                use_batch_norm=False,
                qkv_bias=False,
                linear_bias=head_linear_bias,
                average_pool=head_average_pool,
            )
        else:
            self.head = nn.Identity()


def aimv2_large_native(**kwargs: Any) -> AIMv2VisionEncoder:
    _ = kwargs.pop("img_size", None)
    return AIMv2VisionEncoder(
        patch_size=14,
        embed_dim=1024,
        mlp_hidden_dim=2816,
        num_blocks=24,
        num_heads=8,
        pos_embed_type="sincos",
        head_type=None,
        **kwargs,
    )