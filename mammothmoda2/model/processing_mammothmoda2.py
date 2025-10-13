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


from typing import TYPE_CHECKING, ClassVar, Unpack

import numpy as np
from loguru import logger
from transformers.image_utils import ImageInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLImagesKwargs,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLVideosProcessorKwargs,
)
from transformers.processing_utils import BatchFeature, PreTokenizedInput, ProcessingKwargs, TextInput
from transformers.video_utils import VideoInput

from .mammothmoda2_qwen2_5_vl import MammothUTokenizer

if TYPE_CHECKING:
    from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorFast
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor


DEFAULT_NEGATIVE_PROMPT = (
    "deformed, blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, "
    "extra_limb, ugly, poorly drawn hands, fused fingers, messy drawing, broken legs censor, censored, "
    "censor_bar Create an image from the instruction."
)


class Mammothmoda2ImagesKwargs(Qwen2_5_VLImagesKwargs):
    negative_prompt: str | None
    num_images_per_prompt: int
    cfg_scale: float


class Mammothmoda2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Mammothmoda2ImagesKwargs
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs
    _defaults = {  # noqa: RUF012
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


class Mammothmoda2Processor(Qwen2_5_VLProcessor):
    """The mammothmoda2 processor inherit from Qwen2_5_VLProcessor, adding image editing support."""

    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer", "video_processor"]

    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "MammothUTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer: MammothUTokenizer | None = None,
        video_processor=None,
        chat_template=None,
        t2i_generate: bool = False,
        ar_height: int = 32,
        ar_width: int = 32,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.t2i_generate = t2i_generate
        self.ar_height = ar_height
        self.ar_width = ar_width
        logger.info(f"Mammothmoda2Processor init: {t2i_generate=} | {ar_height=} | {ar_width=}")

        # Type maker for better IDE type hint.
        self.tokenizer: MammothUTokenizer
        self.image_processor: Qwen2VLImageProcessor | Qwen2VLImageProcessorFast
        self.video_processor: Qwen2VLVideoProcessor

    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        **kwargs: Unpack[Mammothmoda2ProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Mammothmoda2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if not isinstance(text, list):
            text = [text]

        # Mammothmoda2 pre-processing: inputs expansion.
        if self.t2i_generate is True:
            num_images_per_prompt = output_kwargs["images_kwargs"]["num_images_per_prompt"]
            cfg_scale = output_kwargs["images_kwargs"]["cfg_scale"]
            if num_images_per_prompt > 1:  # NOTE: num_images_per_prompt > 1, we need to repeat the inputs
                images = images * num_images_per_prompt if images is not None else None
                videos = videos * num_images_per_prompt if videos is not None else None
                text = text * num_images_per_prompt
            if cfg_scale > 1.0:  # NOTE: cfg_scale > 1.0, we need to repeat the inputs
                images = images * 2 if images is not None else None
                videos = videos * 2 if videos is not None else None
                text = text * 2

        # Original Qwen2_5_VLProcessor logic.
        image_inputs = videos_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        if videos is not None:
            fps = output_kwargs["videos_kwargs"].get("fps", 2.0)
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.video_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                error_msg = (
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the "
                    f"length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
                raise ValueError(error_msg)
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        text = text.copy()  # below lines change text in-place
        if images is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    num_video_tokens = video_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.video_token, "<|placeholder|>" * num_video_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        inputs = BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)

        # Mammothmoda2 t2i post-processing: attaching negative prompt.
        if self.t2i_generate is True:
            negative_ids, negative_mask = None, None
            if (negative_prompt := output_kwargs["images_kwargs"].get("negative_prompt", None)) is not None:
                negative_messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful image generator."}]},
                    {"role": "user", "content": [{"type": "text", "text": negative_prompt}]},
                ]
                negative_text = self.apply_chat_template(negative_messages, tokenize=False, add_generation_prompt=False)
                negative_inputs = super().__call__(
                    text=[negative_text] * num_images_per_prompt,
                    images=None,
                    videos=None,
                    return_tensors=return_tensors,
                    padding=True,
                    padding_side="left",
                )
                negative_ids = negative_inputs.input_ids  # [bs, seq_len]
                negative_mask = negative_inputs.attention_mask  # full 1
                inputs["negative_ids"] = negative_ids  # Already Tensor, directly attach.
                inputs["negative_mask"] = negative_mask
        return inputs

    def apply_chat_template(self, *args, **kwargs) -> str:
        if self.t2i_generate is True:  # For t2i, use different chat template.
            kwargs["t2i_generate"] = self.t2i_generate
            kwargs["ar_height"] = self.ar_height
            kwargs["ar_width"] = self.ar_width
        return super().apply_chat_template(*args, **kwargs)
