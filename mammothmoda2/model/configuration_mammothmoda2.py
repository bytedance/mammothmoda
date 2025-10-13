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

from typing import ClassVar, Literal

from transformers import AutoConfig, PretrainedConfig

__all__ = ["Mammothmoda2Config"]


class Mammothmoda2Config(PretrainedConfig):
    """Configuration class for Mammothmoda.

    This is the configuration class to store the configuration of a [`MammothModaModel`]. It is used to
    instantiate a MammothModa model according to the specified arguments, defining the model architecture.
    """

    model_type = "mammothmoda2"
    is_composition = True
    sub_configs: ClassVar = {"llm_config": AutoConfig}

    def __init__(
        self,
        *,
        llm_config: dict | None = None,
        gen_vae_config: dict | None = None,
        gen_dit_config: dict | None = None,
        gen_condition_mode: Literal["text", "image", "text_image"] = "image",
        gen_image_condition_refiner_config: dict | None = None,
        gen_axes_dim_rope: list[int] | None = None,
        gen_axes_lens: list[int] | None = None,
        gen_transport_config: dict | None = None,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        """Initialize the mammothmoda VL model config.

        Args:
            llm_config: The LLM config dict, to initialized the subclass of PretrainedConfig.
            llm_moe_init_gen_from_und: Whether to init the additional generation weights from understanding ones.

            vis_freeze: Whether to freeze the visual model.
            initializer_range: The initializer range/std.
            **kwargs: The kwargs.

        """
        super().__init__(**kwargs)
        self.llm_config = AutoConfig.for_model(**llm_config) if llm_config is not None else None
        self.gen_vae_config = gen_vae_config
        self.gen_dit_config = gen_dit_config

        self.gen_condition_mode = gen_condition_mode
        self.gen_image_condition_refiner_config = gen_image_condition_refiner_config
        self.gen_axes_dim_rope = gen_axes_dim_rope or [40, 40, 40]
        self.gen_axes_lens = gen_axes_lens or [10000, 10000, 10000]
        self.gen_transport_config = gen_transport_config or {}
        self.initializer_range = initializer_range

    def get_text_config(self, decoder=False) -> PretrainedConfig:  # noqa: ARG002, FBT002
        """Pick the text config manually since MammothmodaVLConfig is a irregular one with is_composition==True."""
        return self.llm_config
