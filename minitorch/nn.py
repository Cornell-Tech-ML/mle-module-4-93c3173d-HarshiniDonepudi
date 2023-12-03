from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height, new_width = height // kh, width // kw

    # Reshape and permute
    # First, reshape to separate the kernel dimensions
    new_height = height // kh
    new_width = width // kw

    # Adapting the reshaping and permuting sequence
    output = input.contiguous().view(batch, channel, height, new_width, kw)
    output = output.permute(0, 1, 3, 2, 4)
    output = output.contiguous().view(batch, channel, new_width, new_height, kh * kw)

    return output, new_height, new_width

    # TODO: Implement for Task 4.3.
    # raise NotImplementedError("Need to implement for Task 4.3")


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D

    Args:
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
        Pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.3.
    input, new_height, new_width = tile(input, kernel)
    return input.mean(-1).view(
        batch, channel, new_height, new_width
    )
    # raise NotImplementedError("Need to implement for Task 4.3")


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input : input tensor
        dim : dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        "Forward of max should be max reduction"
        # TODO: Implement for Task 4.4.
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))
        # raise NotImplementedError("Need to implement for Task 4.4")

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim) , 0.0
        # TODO: Implement for Task 4.4.
        # raise NotImplementedError("Need to implement for Task 4.4")


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the softmax as a tensor.



    $z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}$

    Args:
        input : input tensor
        dim : dimension to apply softmax

    Returns:
        softmax tensor
    """
    # TODO: Implement for Task 4.4.
    max_val = max_reduce(input, dim)
    shifted_input = input - max_val
    exps = shifted_input.exp()
    return exps / exps.sum(dim=dim)
    # raise NotImplementedError("Need to implement for Task 4.4")


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    r"""
    Compute the log of the softmax as a tensor.

    $z_i = x_i - \log \sum_i e^{x_i}$

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input : input tensor
        dim : dimension to apply log-softmax

    Returns:
         log of softmax tensor
    """
    # TODO: Implement for Task 4.4
    max_val = max_reduce(input, dim)
    shifted_input = input - max_val
    exps_sum = shifted_input.exp().sum(dim=dim)
    log_exps_sum = exps_sum.log()
    return shifted_input - log_exps_sum

# raise NotImplementedError("Need to implement for Task 4.4")


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled max pooling 2D

    Args:
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
        Tensor : pooled tensor
    """
    batch, channel, height, width = input.shape
    # TODO: Implement for Task 4.4.
    out, new_height, new_width = tile(input, kernel)
    out = max(out, len(out.shape) - 1)
    return out.view(batch, channel, new_height, new_width)
    # raise NotImplementedError("Need to implement for Task 4.4")


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout positions based on random noise.

    Args:
        input : input tensor
        rate : probability [0, 1) of dropping out each position
        ignore : skip dropout, i.e. do nothing at all

    Returns:
        tensor with randoom positions dropped out
    """
    # TODO: Implement for Task 4.4.
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
    # raise NotImplementedError("Need to implement for Task 4.4")
