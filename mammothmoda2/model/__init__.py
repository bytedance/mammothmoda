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

from transformers import AutoProcessor, AutoConfig

from .configuration_mammothmoda2 import Mammothmoda2Config
from .mammothmoda2_qwen3_vl import *  # noqa: F403
from .modeling_mammothmoda2 import Mammothmoda2Model
from .processing_mammothmoda2 import DEFAULT_NEGATIVE_PROMPT, Mammothmoda2Processor

AutoConfig.register(Mammothmoda2Config.model_type, Mammothmoda2Config)
AutoProcessor.register(Mammothmoda2Config, Mammothmoda2Processor)
