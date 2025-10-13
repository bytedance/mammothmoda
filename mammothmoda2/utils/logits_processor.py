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

import torch
from transformers.generation import LogitsProcessor


class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    """
    Logits processor for Classifier-Free Guidance.
    Reference: https://arxiv.org/abs/2207.12598
    """

    def __init__(self, guidance_scale: float) -> None:
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(f"guidance_scale must be > 1, got {guidance_scale}")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores_processed = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        scores_processed = torch.cat([scores_processed, scores_processed], dim=0)
        return scores_processed


class SampledScopeLogitsProcessor(LogitsProcessor):
    """Filters logits to a specific vocabulary range."""

    def __init__(self, scope_start: int, scope_end: int, filter_value: float = -float("Inf")) -> None:
        self.scope_start = scope_start
        self.scope_end = scope_end
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.scope_start:
            scores[:, : self.scope_start] = self.filter_value
        if self.scope_end:
            scores[:, self.scope_end :] = self.filter_value
        return scores
