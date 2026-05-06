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
from torchvision import transforms
from transformers.image_utils import ImageInput
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLImagesKwargs,
    Qwen3VLProcessor,
    Qwen3VLVideosProcessorKwargs,
)
from transformers.processing_utils import BatchFeature, PreTokenizedInput, ProcessingKwargs, TextInput
from transformers.video_utils import VideoInput

from .mammothmoda2_qwen3_vl import MammothUTokenizer

if TYPE_CHECKING:
    from transformers.models.qwen2_vl import Qwen2VLImageProcessor, Qwen2VLImageProcessorFast
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor


DEFAULT_NEGATIVE_PROMPT = (
    "deformed, blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, "
    "extra_limb, ugly, poorly drawn hands, fused fingers, messy drawing, broken legs censor, censored, "
    "censor_bar Create an image from the instruction."
)


num_digits = lambda n: len(str(abs(int(n))))


class Mammothmoda2ImagesKwargs(Qwen3VLImagesKwargs):
    negative_prompt: str | None
    num_images_per_prompt: int
    cfg_scale: float
    t2i_generate: bool


class Mammothmoda2ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Mammothmoda2ImagesKwargs
    videos_kwargs: Qwen3VLVideosProcessorKwargs
    _defaults = {  # noqa: RUF012
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": False,
        },
        "videos_kwargs": {"return_metadata": True},
    }


class Mammothmoda2Processor(Qwen3VLProcessor):
    """The mammothmoda2 processor inherit from Qwen3VLProcessor, adding image editing support."""

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
        gen_patch_size: int = 14,
        gen_image_token: str = "<|gen_placeholder|>",
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)
        self.gen_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.gen_patch_size = gen_patch_size
        self.gen_image_token = gen_image_token

        # Type maker for better IDE type hint.
        self.tokenizer: MammothUTokenizer
        self.image_processor: Qwen2VLImageProcessor | Qwen2VLImageProcessorFast
        self.video_processor: Qwen3VLVideoProcessor

    def __call__(
        self,
        images: ImageInput | None = None,
        gen_images: ImageInput | None = None,
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
        t2i_generate = output_kwargs["images_kwargs"]["t2i_generate"]
        if t2i_generate is True:
            num_images_per_prompt = output_kwargs["images_kwargs"]["num_images_per_prompt"]
            cfg_scale = output_kwargs["images_kwargs"]["cfg_scale"]
            if num_images_per_prompt > 1:  # NOTE: num_images_per_prompt > 1, we need to repeat the inputs
                images = images * num_images_per_prompt if images is not None else None
                gen_images = gen_images * num_images_per_prompt if gen_images is not None else None
                videos = videos * num_images_per_prompt if videos is not None else None
                text = text * num_images_per_prompt
            if cfg_scale > 1.0:  # NOTE: cfg_scale > 1.0, we need to repeat the inputs
                images = images * 2 if images is not None else None
                gen_images = gen_images * 2 if gen_images is not None else None
                videos = videos * 2 if videos is not None else None
                text = text * 2

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None
        
        if gen_images is not None:
            gen_image_inputs = [self.gen_transform(img).unsqueeze(0) for img in gen_images]
            h_patchs = [img.shape[2] // self.gen_patch_size for img in gen_image_inputs]
            w_patchs = [img.shape[3] // self.gen_patch_size for img in gen_image_inputs]
            gen_image_token_nums = [
                h_patch * (w_patch + 1) + 5 + num_digits(h_patch) + num_digits(w_patch)
                for h_patch, w_patch in zip(h_patchs, w_patchs)
            ]
        else:
            gen_image_inputs = None
            gen_image_token_nums = None

        if videos is not None:
            videos_inputs = self.video_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
            # If user has not requested video metadata, pop it
            if "return_metadata" not in kwargs:
                video_metadata = videos_inputs.pop("video_metadata")
            else:
                video_metadata = videos_inputs["video_metadata"]
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None

        text = text.copy()  # below lines change text in-place
        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    num_image_tokens = image_grid_thw[index].prod() // merge_length
                    text[i] = text[i].replace(self.image_token, "<|placeholder|>" * num_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if gen_image_token_nums is not None:
            assert "".join(text).count(self.gen_image_token) == len(gen_image_token_nums), "gen_image_token_nums is not consistent with gen_image in text"
            index = 0
            for i in range(len(text)):
                while self.gen_image_token in text[i]:
                    num_gen_image_tokens = gen_image_token_nums[index]
                    text[i] = text[i].replace(self.gen_image_token, "<|placeholder|>" * num_gen_image_tokens, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.gen_image_token)

        if video_grid_thw is not None:
            merge_length = self.video_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    metadata = video_metadata[index]
                    if metadata.fps is None:
                        logger.warning_once(
                            "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video could not be inferred. "
                            "Probably `video_metadata` was missing from inputs and you passed pre-sampled frames. "
                            "Defaulting to `fps=24`. Please provide `video_metadata` for more accurate results."
                        )
                        metadata.fps = 24 if metadata.fps is None else metadata.fps

                    # if timestamps are not provided, calculate them
                    curr_timestamp = self._calculate_timestamps(
                        metadata.frames_indices,
                        metadata.fps,
                        self.video_processor.merge_size,
                    )

                    video_placeholder = ""
                    frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
                    for frame_idx in range(video_grid_thw[index][0]):
                        curr_time = curr_timestamp[frame_idx]
                        video_placeholder += f"<{curr_time:.1f} seconds>"
                        video_placeholder += (
                            self.vision_start_token + "<|placeholder|>" * frame_seqlen + self.vision_end_token
                        )
                    if f"{self.vision_start_token}{self.video_token}{self.vision_end_token}" in text[i]:
                        text[i] = text[i].replace(
                            f"{self.vision_start_token}{self.video_token}{self.vision_end_token}", video_placeholder, 1
                        )
                    else:
                        # vllm may input video token directly
                        text[i] = text[i].replace(self.video_token, video_placeholder, 1)
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
        if t2i_generate is True:
            inputs["gen_pixel_values"] = gen_image_inputs
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

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        *args,
        t2i_generate: bool = False,
        ar_height: int = 16,
        ar_width: int = 16,
        **kwargs
    ) -> str:
        if t2i_generate is True:  # For t2i, use different chat template.
            kwargs["t2i_generate"] = t2i_generate
            kwargs["ar_height"] = ar_height
            kwargs["ar_width"] = ar_width
        return super().apply_chat_template(conversation, *args, **kwargs)
