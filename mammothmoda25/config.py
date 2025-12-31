import dataclasses
from dataclasses import dataclass, asdict, field 
from typing import Dict, Any, Optional

import torch

@dataclass
class MoELayerConfig():
    num_layers: int = None
    hidden_size: int = None
    num_attention_heads: int = None
    ffn_hidden_size: int = None
    num_moe_experts: int = None
    moe_router_score_function: str = None
    moe_router_enable_expert_bias: bool = True
    moe_grouped_gemm: bool = False
    moe_router_topk: int = 4
    moe_router_dtype: str = "fp32"
    add_bias_linear: bool = False
    use_cpu_initialization: bool = True
    gated_linear_unit: bool = False
    params_dtype: torch.dtype = torch.bfloat16
    num_shared_experts: int = 0
    shared_expert_intermediate_size: int = 2048


@dataclass
class TransformerConnectorConfig:
    hidden_size: int = 4096
    num_heads: int = 8
    ffn_dim: int = 4096
    dropout: float = 0.0
    attention_dropout: float = 0.0
    norm_type: str = "rmsnorm"
    eps: float = 1e-6

@dataclass
class VLMTokenRefinerConfig:
    # 顶层配置参数
    mode: str = "ffn_with_bidirectional_transformer"
    fusion_mode: str = "in_context"
    num_transformer_layers: int = 2
    use_ffn: bool = True
    use_transformer: bool = True
    vlm_dim: int = 4096
    transformer_config: TransformerConnectorConfig = field(default_factory=TransformerConnectorConfig)

    def to_dict(self) -> Dict[str, Any]:
        config_dict = asdict(self)
        return config_dict