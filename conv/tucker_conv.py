import torch
import math
from conv import decompositions

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
    
        self.filter = torch.Tensor(
            self.out_channels, 
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        )   
        self.filter = torch.nn.Parameter(self.filter)
        torch.nn.init.kaiming_uniform_(self.filter, a=math.sqrt(5))
        self.layer = torch.nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = self.kernel_size,
            stride = self.stride, 
        )
        rank = rank = max(self.layer.weight.data.numpy().shape)//3
        self.layer = decompositions.tucker_decomposition_conv_layer(
            self.layer,
        )

    def calculate_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.out_channels,
            int((input_shape[-2]-self.kernel_size[0])/self.stride[0] + 1), 
            int((input_shape[-1]-self.kernel_size[1])/self.stride[1] + 1)
        )   

    def forward(self, x):
        return self.layer(x)

