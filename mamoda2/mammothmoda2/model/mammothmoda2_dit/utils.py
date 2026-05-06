import torch
from einops import rearrange, repeat


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)).reshape(
            -1, *other_shape
        )

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


# Copied from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply