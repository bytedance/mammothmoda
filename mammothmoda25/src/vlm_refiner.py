"""
Connector module for VLM to DiT feature transformation.
Supports various architectures including FFN-only, FFN + Transformer layers, etc.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from diffusers.models.normalization import RMSNorm


class TransformerLayer(nn.Module):
    """
    Bidirectional Transformer layer for connector.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Pre-norm
        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(dim=hidden_size, eps=eps)
            self.norm2 = RMSNorm(dim=hidden_size, eps=eps)
        else:
            self.norm1 = nn.LayerNorm(hidden_size, eps=eps)
            self.norm2 = nn.LayerNorm(hidden_size, eps=eps)
        
        # Multi-head attention (bidirectional)
        self.self_attn = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
            attention_mask: Optional attention mask
        Returns:
            Output tensor with same shape as input
        """
        # Ensure input is 3D [batch, seq, hidden]
        original_shape = x.shape
        if x.dim() == 4:
            # If 4D (e.g., [batch, 1, seq, hidden]), squeeze the extra dim
            x = x.squeeze(1)
        elif x.dim() == 2:
            # If 2D, add batch dimension
            x = x.unsqueeze(0)
        
        x = x.bfloat16()
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = residual + attn_output
        
        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        # Restore original shape if needed
        if len(original_shape) == 4 and original_shape[1] == 1:
            x = x.unsqueeze(1)
        elif len(original_shape) == 2:
            x = x.squeeze(0)
        
        return x


class VLMTokenRefiner(nn.Module):
    """
    Configurable connector between VLM encoder and DiT predictor.
    
    Supports multiple modes:
    - ffn_only: Simple FFN projection (backward compatible)
    - ffn_with_bidirectional_transformer: FFN + multiple bidirectional transformer layers
    
    Additionally supports different fusion modes via the `fusion_mode` config:
    - cross_attn (default): VLM features interact with DiT via cross-attention
    - in_context: VLM features are concatenated to DiT input for self-attention
    
    Args:
        in_dim: Input dimension from VLM encoder
        out_dim: Output dimension to DiT predictor  
        connector_config: Configuration dict with mode-specific parameters
            - mode: "ffn_only" or "ffn_with_bidirectional_transformer"
            - fusion_mode: "cross_attn" (default) or "in_context"
            - transformer_config: (for ffn_with_bidirectional_transformer mode)
                - hidden_size: Hidden size for transformer layers
                - num_heads: Number of attention heads
                - ffn_dim: FFN hidden dimension
                - dropout: Dropout rate
                - attention_dropout: Attention dropout rate
                - norm_type: "rmsnorm" or "layernorm"
                - eps: Normalization epsilon
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        connector_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Default to backward-compatible FFN-only mode
        if connector_config is None:
            connector_config = {"mode": "ffn_only"}
        
        self.mode = connector_config.get("mode", "ffn_only")
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Build connector based on mode
        if self.mode == "ffn_only":
            # Backward compatible: simple FFN
            self.connector = self._build_ffn_only(in_dim, out_dim)
        
        elif self.mode == "ffn_with_bidirectional_transformer":
            # New mode: FFN + bidirectional transformer layers
            self.connector = self._build_ffn_with_transformer(
                in_dim, out_dim, connector_config
            )
        
        else:
            raise ValueError(f"Unknown connector mode: {self.mode}")
    
    def _build_ffn_only(self, in_dim: int, out_dim: int) -> nn.Module:
        """Build simple FFN connector (backward compatible)."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    
    def _build_ffn_with_transformer(
        self,
        in_dim: int,
        out_dim: int,
        config: Dict[str, Any],
    ) -> nn.Module:
        """Build FFN + bidirectional transformer connector."""
        transformer_config = config.get("transformer_config", {})
        num_layers = config.get("num_transformer_layers", 2)
        
        # Get transformer layer config
        hidden_size = transformer_config.get("hidden_size", out_dim)
        num_heads = transformer_config.get("num_heads", 32)
        ffn_dim = transformer_config.get("ffn_dim", hidden_size * 4)
        dropout = transformer_config.get("dropout", 0.0)
        attention_dropout = transformer_config.get("attention_dropout", 0.0)
        norm_type = transformer_config.get("norm_type", "rmsnorm")
        eps = transformer_config.get("eps", 1e-6)
        
        layers = []
        
        # Input projection FFN
        if config.get("use_ffn", True):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        else:
            # Direct projection if FFN is disabled
            if in_dim != hidden_size:
                layers.append(nn.Linear(in_dim, hidden_size))
        
        # Bidirectional transformer layers
        if config.get("use_transformer", True) and num_layers > 0:
            for _ in range(num_layers):
                layers.append(
                    TransformerLayer(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        norm_type=norm_type,
                        eps=eps,
                    )
                )
        
        # Output projection if dimensions don't match
        if hidden_size != out_dim:
            layers.append(nn.Linear(hidden_size, out_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, in_dim]
        Returns:
            Output tensor [batch_size, seq_len, out_dim]
        """
        return self.connector(x)

