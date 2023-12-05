from typing import Tuple

import numpy as np
from numba import njit, prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    # Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
to_index = njit(inline="always")(to_index)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Convolution implementation.

    Given input tensor of

       `batch, in_channels, width`

    and weight tensor

       `out_channels, in_channels, k_width`

    Computes padded output of

       `batch, out_channels, width`

    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)

    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides

    # Pre-calculate stride values that are constant within the inner loops
    out_stride0, out_stride1, out_stride2 = out_strides
    # Loop over the output tensor's elements.
    for i in prange(out_size):
        # Convert the linear index to a multi-dimensional index.
        out_index = np.empty(MAX_DIMS, np.int32)
        to_index(i, out_shape, out_index)
        cur_batch, curr_out, curr_w = out_index[:len(out_shape)]

        # Calculate the linear position in the output tensor.
        out_pos = cur_batch * out_stride0 + curr_out * out_stride1 + curr_w * out_stride2

        # Loop over the input channels and kernel width for convolution.
        for curr_channel in prange(in_channels):
            for curr_kw in range(kw):
                # Handling reverse anchoring of the kernel.
                adjusted_kw = kw - curr_kw - 1 if reverse else curr_kw

                # Calculate the linear position in the weight tensor.
                weight_idx = curr_out * s2[0] + curr_channel * s2[1] + adjusted_kw * s2[2]
                accum = 0.0

                # Adjusted kernel position based on 'reverse'
                kernel_pos = curr_w - adjusted_kw if reverse else curr_w + adjusted_kw

                # Perform the convolution operation based on the 'reverse' flag.
                if 0 <= kernel_pos < width:
                    input_idx = cur_batch * s1[0] + curr_channel * s1[1] + kernel_pos * s1[2]
                    accum = input[input_idx]

                # Update the output tensor at the calculated position.
                out[out_pos] += accum * weight[weight_idx]

# TODO: Implement for Task 4.1.
# raise NotImplementedError("Need to implement for Task 4.1")


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight : out_channel x in_channel x kh x kw

        Returns:
            batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
    (See diagrams)


    Args:
        out (Storage): storage for `out` tensor.
        out_shape (Shape): shape for `out` tensor.
        out_strides (Strides): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (Storage): storage for `input` tensor.
        input_shape (Shape): shape for `input` tensor.
        input_strides (Strides): strides for `input` tensor.
        weight (Storage): storage for `input` tensor.
        weight_shape (Shape): shape for `input` tensor.
        weight_strides (Strides): strides for `input` tensor.
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]
    out_stride0, out_stride1, out_stride2, out_stride3 = out_strides
    # TODO: Implement for Task 4.2.
    for i in prange(out_size):
        # Convert the linear index to a multi-dimensional index.
        out_index = np.empty(MAX_DIMS, np.int32)
        to_index(i, out_shape, out_index)
        cur_batch, curr_out, curr_h, curr_w = out_index[:len(out_shape)]

        # Calculate the linear position in the output tensor.
        out_pos = (cur_batch * out_stride0 + curr_out * out_stride1 + curr_h * out_stride2 + curr_w * out_stride3)

        # Loop over the input channels and kernel dimensions for convolution.
        for curr_channel in prange(in_channels):
            for curr_kh in range(kh):
                for curr_kw in range(kw):
                    # Handling reverse anchoring of the kernel.
                    adjusted_kh = kh - curr_kh - 1 if reverse else curr_kh
                    adjusted_kw = kw - curr_kw - 1 if reverse else curr_kw

                    # Calculate the linear positions in the weight tensor.
                    weight_idx = (curr_out * s20 + curr_channel * s21 + adjusted_kh * s22 + adjusted_kw * s23)
                    accum = 0.0

                    # Adjusted kernel position based on 'reverse'
                    kernel_h_pos = curr_h - adjusted_kh if reverse else curr_h + adjusted_kh
                    kernel_w_pos = curr_w - adjusted_kw if reverse else curr_w + adjusted_kw

                    # Perform the convolution operation based on the 'reverse' flag.
                    if 0 <= kernel_h_pos < height and 0 <= kernel_w_pos < width:
                        input_idx = (cur_batch * s10 + curr_channel * s11 + kernel_h_pos * s12 + kernel_w_pos * s13)
                        accum = input[input_idx]

                    # Update the output tensor at the calculated position.
                    out[out_pos] += accum * weight[weight_idx]

# raise NotImplementedError("Need to implement for Task 4.2")


tensor_conv2d = njit(parallel=True, fastmath=True)(_tensor_conv2d)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
