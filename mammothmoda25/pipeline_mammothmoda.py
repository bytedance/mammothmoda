from typing import Optional
from diffusers import WanPipeline
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel 
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


class MammothModa25Pipeline(WanPipeline):

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
        # Use qwen3vl as text encoder
        self.text_encoder = self.text_encoder.model
