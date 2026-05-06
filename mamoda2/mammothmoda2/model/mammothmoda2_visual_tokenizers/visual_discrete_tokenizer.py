import math
import traceback

import numpy as np
import torch
from addict import Dict as ADict
from loguru import logger
from PIL import Image
from torch import nn
from torchvision import transforms

from .mammothtok import VQVitModel2DPlus_AIMv2, VQVitModel2DPlusArgs


class VisualDiscreteTokenizer(nn.Module):
    _is_hf_initialized = True

    def __init__(self, config, device=None) -> None:
        super().__init__()
        self.config = config
        self._device = device

        self.image_size = self.config.image_size
        self.downsample_ratio = 14
        self.upsample_ratio = 14
        self.image_tokenizer = VQVitModel2DPlus_AIMv2(
            VQVitModel2DPlusArgs(
                model_size=None,
                encoder_size=None,
                decoder_size="large",
                adaptive_gn=True,
                d2s_up=True,
                rot=True,
                use_attn=False,
                distill_depth=3,
                use_rope=True,
                image_size=self.config.image_size,
                fea_rec_loss_type="mse",
                fea_rec_loss_weight=1.0,
                transformer_layer_type="TransformerLayer",
                z_channels=1024,
                out_inner_dim=1536,
                up_sample_ratio=14,
            )
        )
        self.spatial_scale_factor = config.spatial_scale_factor
        logger.info(f"Initializing VisualDiscreteTokenizer with config: {self.config}")

    @property
    def device(self) -> torch.device:
        return self._device or next(self.parameters()).device

    def _preprocess_image(self, image_input):
        if isinstance(image_input, (str, Image.Image)):
            return [image_input]
        return image_input

    def _format_output(self, tokens, dtype=np.uint16):
        return [t.cpu().numpy().astype(dtype) for t in tokens.split(1, dim=0)]

    def forward(
        self,
        x: torch.Tensor,
        return_tokens: bool = True,
        return_tensor: bool = False,
    ):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            return_tokens (bool, optional): Whether to return tokens. Defaults to True.
            return_tensor (bool, optional): Whether to return tensor. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        x = x.to(self.device)
        b_x, h_x, w_x = x.shape[0], x.shape[2], x.shape[3]
        quant, _, (_, _, indices) = self.image_tokenizer.encode(x)

        # 直接返回 tokenizer 编码的 indices
        if return_tensor:
            return indices

        # 返回适配后续 tokens -> tokens str 的 function tokens
        if return_tokens:
            # 重新组织索引为图像形式 适配现有的 tokens -> tokens str 的 function
            h_tokens = h_x // self.downsample_ratio
            w_tokens = w_x // self.downsample_ratio
            indices = indices.reshape(b_x, h_tokens, w_tokens, 1)
            tokens = []
            for b in range(b_x):
                token_info = {
                    "global": indices[b].cpu().numpy().astype(np.uint16),
                }
                tokens.append(token_info)
            return tokens

        # 使用模型解码生成图像
        samples = self.image_tokenizer.decode_code(indices, shape=indices.shape)
        # 将值从[-1,1]范围转换为[0,255]的RGB图像
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        return samples


def get_mammothmoda2_visual_tokenizer(mode: str, device: str = None, dtype=torch.bfloat16) -> VisualDiscreteTokenizer:
    """
    Construct and return a MammothUProcessor instance from the given root_dir.
    If root_dir is None, use the default path.
    """
    visual_tokenizer = VisualDiscreteTokenizer(
        config=ADict(
            mode="mammothtok_aimv2",
            image_size=448,
            codebook_size=1024 * 32,
            spatial_scale_factor=1,
        ),
        device=device,
    )
    return visual_tokenizer