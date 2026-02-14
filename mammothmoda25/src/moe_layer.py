# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch


class MoELayer(torch.nn.Module):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(self, config):
        super().__init__()
        self.router = TopKRouter(config=config)
        self.experts = Expert(config.num_moe_experts, config, is_shared=False)
        if config.num_shared_experts > 0:
            self.shared_experts = Expert(config.num_shared_experts, config, is_shared=True)
        else:
            self.shared_experts = None


    def forward(self, hidden_states: torch.Tensor):

        def custom_forward(hidden_states):
            probs, routing_map = self.router(hidden_states)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            output_flat = torch.zeros_like(flat_hidden)
            num_experts = self.experts.num_local_experts

            w1 = self.experts.weight1.view(num_experts, hidden_dim, -1)
            w2 = self.experts.weight2.view(num_experts, -1, hidden_dim)

            for expert_idx in range(num_experts):
                token_mask = torch.isin(routing_map, expert_idx).any(dim=-1)
                if not token_mask.any():
                    continue

                expert_input = flat_hidden[token_mask]
                routing_map_subset = routing_map[token_mask]
                probs_subset = probs[token_mask]
                expert_pos_mask = (routing_map_subset == expert_idx)
                expert_weights = (probs_subset * expert_pos_mask).sum(dim=-1, keepdim=True)
                w1_single = w1[expert_idx]
                w2_single = w2[expert_idx]

                fc1_output = torch.matmul(expert_input, w1_single)
                intermediate_parallel = self.experts.activation_func(fc1_output)
                expert_output = torch.matmul(intermediate_parallel, w2_single)

                weighted_output = expert_output * expert_weights
                output_flat[token_mask] += weighted_output

            
            if self.shared_experts is not None:
                
                shared_fc1_out = torch.matmul(flat_hidden, self.shared_experts.weight1.t())
                shared_inter = self.shared_experts.activation_func(shared_fc1_out)
                shared_out = torch.matmul(shared_inter, self.shared_experts.weight2.t())
                output_flat += shared_out
                
            mlp_bias = None
            output = output_flat.reshape(batch_size, seq_len, hidden_dim)

            return output, mlp_bias
        
        output, mlp_bias = custom_forward(hidden_states)

        return output


class TopKRouter(torch.nn.Module):
    """Route each token to the top-k experts."""

    def __init__(
        self,
        config,
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__()
        self.config = config
        self.topk = self.config.moe_router_topk
        self.score_function = self.config.moe_router_score_function
        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(
            torch.zeros((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
        )
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)

        self.enable_expert_bias = self.config.moe_router_enable_expert_bias
        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.int64),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None


    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # A naive top-k routing without load balancing
        scores, indices, tokens_per_expert = self.topk_softmax_with_capacity(
            logits,
            self.topk,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        ) 
        return scores, indices

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]
        self._maintain_float32_expert_bias()

        logits = self.gating(input)
        logits = logits.view(-1, self.config.num_moe_experts)

        scores, indices = self.routing(logits)

        return scores, indices
    
    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        router_dtype = input.dtype
        if self.config.moe_router_dtype == 'fp32':
            router_dtype = torch.float32
        elif self.config.moe_router_dtype == 'fp64':
            router_dtype = torch.float64
        logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))
        return logits
    
    def topk_softmax_with_capacity(
        self,
        logits: torch.Tensor,
        topk: int,
        score_function: str = "sigmoid",
        expert_bias: Optional[torch.Tensor] = None,
        use_pre_softmax: bool = False
    ):

        assert logits.dim() == 2, f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}."
        num_tokens = logits.shape[0]
        num_experts = logits.shape[1]


        def compute_topk(scores, topk):
            return torch.topk(scores, k=topk, dim=1)

        if score_function == "softmax":
            if use_pre_softmax:
                scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
                probs, top_indices = compute_topk(scores, topk)
            else:
                scores, top_indices = compute_topk(logits, topk)
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)
        elif score_function == "sigmoid":
            scores = torch.sigmoid(logits.float()).type_as(logits)
            if expert_bias is not None:
                scores_for_routing = scores + expert_bias
                _, top_indices = compute_topk(scores_for_routing, topk)
                scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
            else:
                scores, top_indices = compute_topk(scores, topk)
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
        else:
            raise ValueError(f"Invalid score_function: {score_function}")

        # TopK without capacity
        tokens_per_expert = torch.bincount(top_indices.view(-1), minlength=num_experts)
        return probs, top_indices, tokens_per_expert

    def _maintain_float32_expert_bias(self):
        """
        Maintain the expert bias in float32.

        When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
        We keep it in float32 to avoid routing errors when updating the expert_bias.
        """
                
        if hasattr(self, 'expert_bias') and self.expert_bias is not None:
            if self.expert_bias.dtype != torch.float32:
                self.expert_bias.data = self.expert_bias.data.to(torch.float32)


class Expert(torch.nn.Module):
    """Grouped MLP layer.

    Args:
        num_experts (int): Number of experts.
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, num_experts: int, config, is_shared: bool = False):
        super().__init__()
        self.config = config
        self.num_local_experts = num_experts
        if is_shared:
            fc1_output_size = self.config.shared_expert_intermediate_size * self.num_local_experts
        else:
            fc1_output_size = self.config.ffn_hidden_size * self.num_local_experts
        fc2_input_size = fc1_output_size

        if config.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2

        self.activation_func = torch.nn.GELU(approximate="tanh")
        if is_shared:
            self.weight1 = torch.nn.Parameter(
                torch.zeros(
                    fc1_output_size,
                    self.config.hidden_size,
                    dtype=torch.bfloat16,
                )
            )
            self.weight2 = torch.nn.Parameter(
                torch.zeros(
                    self.config.hidden_size,
                    fc2_input_size,
                    dtype=torch.bfloat16,
                )
            )
        else:
            self.weight1 = torch.nn.Parameter(
                torch.zeros(
                    self.config.hidden_size,
                    fc1_output_size,
                    dtype=torch.bfloat16,
                )
            )
            self.weight2 = torch.nn.Parameter(
                torch.zeros(
                    fc2_input_size,
                    self.config.hidden_size,
                    dtype=torch.bfloat16,
                )
            )



    