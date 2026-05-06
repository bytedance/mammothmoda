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

import random
import re
from typing import Self

import numpy as np
import torch
from accelerate.utils import is_npu_available

__all__ = ["Singleton", "remap_unified_tokens", "set_seed"]


def set_seed(seed) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if is_npu_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def remap_unified_tokens(generated_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """Remaps generated string-based visual tokens back to integer token IDs."""
    unified_tokens_str = tokenizer.decode(generated_ids)
    visual_template_regex = r"<\|visual token (\d+)\|>"
    token_ids = re.findall(visual_template_regex, unified_tokens_str)
    return torch.tensor([int(m) for m in token_ids], dtype=torch.long)


class Singleton:
    """Singleton base class."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs) -> Self:  # noqa: ARG004, allow args and kwargs for __init__
        """Override __new__ method to ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
