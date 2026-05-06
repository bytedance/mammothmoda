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


"""Init mammothmoda_qwen3_vl model."""

from transformers import AutoConfig, AutoModelForVision2Seq, AutoTokenizer

from .configuration_mammothmoda2_qwen3_vl import Mammothmoda2Qwen3VLConfig
from .modeling_mammothmoda2_qwen3_vl import (
    Mammothmoda2Qwen3VLCausalLMOutputWithPast,
    Mammothmoda2Qwen3VLForConditionalGeneration,
)
from .tokenization_mammothmoda2_qwen3_vl import MammothUTokenizer

# Huggingface AutoClass register.
AutoConfig.register(Mammothmoda2Qwen3VLConfig.model_type, Mammothmoda2Qwen3VLConfig)
AutoModelForVision2Seq.register(Mammothmoda2Qwen3VLConfig, Mammothmoda2Qwen3VLForConditionalGeneration)
AutoTokenizer.register(
    config_class=Mammothmoda2Qwen3VLConfig,
    slow_tokenizer_class=MammothUTokenizer,
)


__all__ = [
    "MammothUTokenizer",
    "Mammothmoda2Qwen3VLCausalLMOutputWithPast",
    "Mammothmoda2Qwen3VLConfig",
    "Mammothmoda2Qwen3VLForConditionalGeneration",
]
