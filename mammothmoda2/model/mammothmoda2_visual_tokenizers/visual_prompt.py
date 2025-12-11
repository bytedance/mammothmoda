import numpy as np
import torch

visual_template = ("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>")
prefix_template = "{H}*{W}"


def to_imgstr(image_tokens, tokenizer):
    if not isinstance(image_tokens, list):
        if isinstance(image_tokens, torch.Tensor):
            image_tokens = image_tokens.cpu().numpy().tolist()
        else:
            image_tokens = image_tokens.tolist()

    image_token_str = [
        [visual_template[0].format(token_id=token_id) for token_id in token_row] for token_row in image_tokens
    ]
    image_row_str = ["".join(token_row) for token_row in image_token_str]
    imgstr = tokenizer.eol_token.join(image_row_str)
    return imgstr


def to_imgstr_flatten(image_tokens):
    if not isinstance(image_tokens, list):
        if isinstance(image_tokens, torch.Tensor):
            image_tokens = image_tokens.cpu().numpy().tolist()
        else:
            image_tokens = image_tokens.tolist()

    image_token_str = [visual_template[0].format(token_id=tok) for tok in image_tokens]
    imgstr = "".join(image_token_str)
    return imgstr


def token_reformat(image_tokens, h=4, w=4):
    """
    Reformats image tokens to a new shape (h, w, h * w)
    """
    bs, h_input, w_input = image_tokens.shape

    assert bs == 1

    if h_input % h != 0:
        h_input = h_input // h * h
    if w_input % w != 0:
        w_input = w_input // w * w

    image_tokens = image_tokens[:, :h_input, :w_input]

    return image_tokens


def token_folding(image_tokens, h=2, w=8):
    """
    Folds image tokens into a more compact representation with specified dimensions.

    Args:
        image_tokens (numpy.ndarray): Input image tokens with shape (batch_size, height, width)
        h (int, optional): Height of each folded block. Defaults to 2.
        w (int, optional): Width of each folded block. Defaults to 8.

    Returns:
        numpy.ndarray: Folded tokens with shape (h_input//h, w_input//w, h*w)
    """

    bs, h_input, w_input = image_tokens.shape

    assert bs == 1

    if h_input % h != 0:
        h_input = h_input // h * h
    if w_input % w != 0:
        w_input = w_input // w * w

    image_tokens = image_tokens[:, :h_input, :w_input]
    image_tokens = image_tokens.reshape(bs, h_input // h, h, w_input // w, w)
    image_tokens = np.transpose(image_tokens, (0, 1, 3, 2, 4))
    image_tokens = image_tokens.reshape(h_input // h, w_input // w, h * w)

    return image_tokens


def create_image_prompt(image_tokens, tokenizer, mode="movqgan_2x8", imgstr_mode="naive", return_hw=False):
    """
    Creates an image prompt string based on the given image tokens and processing mode.
    Args:
    Returns:
        str: The formatted image prompt string.
    """

    def _create_image_str(view, tag, imgstr_mode="naive"):
        """
        Helper function to create image prompt for a single view.

        Args:
            view (numpy.ndarray): The image tokens for a single view.
            tag (str): The tag to prepend (e.g., 'Global', 'Highres').
            imgstr_mode (str, optional): The image string mode. Defaults to 'naive'.

        Returns:
            str: The formatted image prompt for the view.
        """
        if imgstr_mode == "naive":
            imgstr = to_imgstr_flatten(view.reshape(-1))

            return (tokenizer.boi_token + tag + tokenizer.img_token + imgstr + tokenizer.eoi_token), (h, w)

        elif imgstr_mode == "split_row":
            assert len(view.shape) == 3, "Expected view to have 3 dimensions for 'split_row' mode."

            h, w, c = view.shape
            imgstr = "".join([to_imgstr_flatten(view[i].reshape(-1)) + tokenizer.eol_token for i in range(h)])
            if c == 1:
                imgstr_compare = to_imgstr(view.squeeze(-1), tokenizer)
                assert imgstr[: len(imgstr_compare)] == imgstr_compare
                imgstr = imgstr_compare

            return (
                tokenizer.boi_token
                + prefix_template.format(H=h, W=w)
                + tokenizer.img_token
                + imgstr
                + tokenizer.eol_token
                + tokenizer.eof_token
                + tokenizer.eoi_token
            ), (h, w)
        else:
            raise ValueError(f"imgstr_mode '{imgstr_mode}' is not supported.")

    if "global" in mode:
        if not isinstance(image_tokens, dict):
            # TODO: fix this
            global_view = image_tokens.reshape(32, 32, -1)
        else:
            global_view = image_tokens.get("global")

        global_image_prompt, hw = _create_image_str(global_view, "Global", imgstr_mode)
        image_prompt = f"{global_image_prompt}"

    else:
        if not isinstance(image_tokens, dict):
            raise TypeError("Expected image_tokens to be a dict for 'global_local' mode.")
        global_view = image_tokens.get("global")
        local_view = image_tokens.get("local")
        if global_view is None or local_view is None:
            raise KeyError("'global' or 'local' key not found in image_tokens for 'global_local' mode.")
        global_image_prompt, _ = _create_image_str(global_view, "Global", imgstr_mode)
        local_image_prompt, _ = _create_image_str(local_view, "Highres", imgstr_mode)
        image_prompt = f"{global_image_prompt}{local_image_prompt}"
        hw = None

    if return_hw:
        return image_prompt, hw
    else:
        return image_prompt


def create_image_prompt_batch(image_tokens_list, tokenizer, mode="movqgan_2x8", imgstr_mode="naive", return_hw=False):
    """
    Creates an image prompt string based on the given image tokens and processing mode.
    Args:
    Returns:
        str: The formatted image prompt string.
    """
    image_prompt_list, hw_list = [], []
    for image_tokens in image_tokens_list:
        image_prompt, hw = create_image_prompt(image_tokens, tokenizer, mode, imgstr_mode, return_hw)
        image_prompt_list.append(image_prompt)
        hw_list.append(hw)

    return image_prompt_list, hw_list
