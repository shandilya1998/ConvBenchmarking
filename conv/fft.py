import math
import torch
from functools import partial
from typing import Tuple, Union, Iterable
from torch import rfftn, irfftn
import torch.nn.functional as f

def complex_matmul(a: torch.Tensor, b: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """Multiplies two complex-valued tensors."""
    # Scalar matrix multiplication of two tensors, over only the first channel
    # dimensions. Dimensions 3 and higher will have the same shape after multiplication.
    # We also allow for "grouped" multiplications, where multiple sections of channels
    # are multiplied independently of one another (required for group convolutions).
    scalar_matmul = partial(torch.einsum, "agc..., gbc... -> agb...")
    a = a.view(a.size(0), groups, -1, *a.shape[2:])
    b = b.view(groups, -1, *b.shape[1:])

    # Compute the real and imaginary parts independently, then manually insert them
    # into the output Tensor.  This is fairly hacky but necessary for PyTorch 1.7.0,
    # because Autograd is not enabled for complex matrix operations yet.  Not exactly
    # idiomatic PyTorch code, but it should work for all future versions (>= 1.7.0).
    real = scalar_matmul(a.real, b.real) - scalar_matmul(a.imag, b.imag)
    imag = scalar_matmul(a.imag, b.real) + scalar_matmul(a.real, b.imag)
    c = torch.zeros(real.shape, dtype=torch.complex64, device=a.device)
    c.real, c.imag = real, imag

    return c.view(c.size(0), -1, *c.shape[3:])


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the
    padding and stride for convolutions, where users may only provide an integer.
    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple
    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


def fft_conv(
    signal: torch.Tensor,
    kernel: torch.Tensor,
    bias: torch.Tensor = None,
    padding: Union[int, Iterable[int]] = 0,
    stride: Union[int, Iterable[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Performs N-d convolution of Tensors using a fast fourier transform, which
    is very fast for large kernel sizes. Also, optionally adds a bias Tensor after
    the convolution (in order ot mimic the PyTorch direct convolution).
    Args:
        signal: (Tensor) Input tensor to be convolved with the kernel.
        kernel: (Tensor) Convolution kernel.
        bias: (Tensor) Bias tensor to add to the output.
        padding: (Union[int, Iterable[int]) Number of zero samples to pad the
            input on the last dimension.
        stride: (Union[int, Iterable[int]) Stride size for computing output values.
    Returns:
        (Tensor) Convolved tensor
    """
    # Cast padding & stride to tuples.
    padding_ = to_ntuple(padding, n=signal.ndim - 2)
    stride_ = to_ntuple(stride, n=signal.ndim - 2)

    # Pad the input signal & kernel tensors
    signal_padding = [p for p in padding_[::-1] for _ in range(2)]
    signal = torch.nn.functional.pad(signal, signal_padding)

    # Because PyTorch computes a *one-sided* FFT, we need the final dimension to
    # have *even* length.  Just pad with one more zero if the final dimension is odd.
    if signal.size(-1) % 2 != 0:
        signal_ = torch.nn.functional.pad(signal, [0, 1])
    else:
        signal_ = signal

    kernel_padding = [
        pad
        for i in reversed(range(2, signal_.ndim))
        for pad in [0, signal_.size(i) - kernel.size(i)]
    ]
    padded_kernel = torch.nn.functional.pad(kernel, kernel_padding)

    # Perform fourier convolution -- FFT, matrix multiply, then IFFT
    # signal_ = signal_.reshape(signal_.size(0), groups, -1, *signal_.shape[2:])
    signal_fr = rfftn(signal_, dim=tuple(range(2, signal.ndim)))
    kernel_fr = rfftn(padded_kernel, dim=tuple(range(2, signal.ndim)))

    kernel_fr.imag *= -1
    output_fr = complex_matmul(signal_fr, kernel_fr, groups=groups)
    output = irfftn(output_fr, dim=tuple(range(2, signal.ndim)))

    # Remove extra padded values
    crop_slices = [slice(0, output.size(0)), slice(0, output.size(1))] + [
        slice(0, (signal.size(i) - kernel.size(i) + 1), stride_[i - 2])
        for i in range(2, signal.ndim)
    ]
    output = output[crop_slices].contiguous()

    # Optionally, add a bias term before returning.
    if bias is not None:
        bias_shape = tuple([1, -1] + (signal.ndim - 2) * [1])
        output += bias.view(bias_shape)

    return output

class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias = True,
        bound = 5,
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise ValueError('Kernel size must be an int or a tuple got {t}'.format(t = type(kernel_size)))

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise ValueError('Kernel size must be an int or a tuple got {t}'.format(t = type(kernel_size)))
        self.use_bias = bias
        if bias:
            self.bias = torch.Tensor(self.out_channels)
            self.bias = torch.nn.Parameter(self.bias)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None    

        self.filter = torch.Tensor(
            self.out_channels,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )
        self.filter = torch.nn.Parameter(self.filter)
        torch.nn.init.kaiming_uniform_(self.filter, a=math.sqrt(5))

    def calculate_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.out_channels,
            int((input_shape[-2]-self.kernel_size[0])/self.stride[0] + 1),
            int((input_shape[-1]-self.kernel_size[1])/self.stride[1] + 1)
        )

    def forward(self, x):
        return fft_conv(
            x,
            self.filter,
            bias=self.bias,
            stride=self.stride,
        )

