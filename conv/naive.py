import torch
import math
from tqdm import tqdm

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
            raise(
                ValueError(
                    'Kernel size must be an int or a tuple got {t}'.format(
                        t = type(kernel_size)
                    )
                )
            )
        
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise(
                ValueError(
                    'Kernel size must be an int or a tuple got {t}'.format(
                        t = type(kernel_size)
                    )
                )
            )
        self.use_bias = bias
        if bias:
            self.bias = torch.Tensor(self.out_channels)
            self.bias = torch.nn.Parameter(self.bias)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = torch.zeros(self.out_channels)        

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

    def forward(self, inp):
        bs, _, h_out, w_out = self.calculate_output_shape(inp.shape)
        self.out = torch.zeros(bs, self.out_channels, h_out, w_out)

        if inp.shape[1] != self.in_channels:
            raise(
                ValueError(
                    'Expected Number of channels in input is `{i}`, got `{got}`'.format(
                        i = self.in_channels, 
                        got = inp.shape[1]
                    )
                )
            )

        for n in range(bs):
            for m in range(self.out_channels):
                for x in range(w_out):
                    for y in range(h_out):
                        self.out[n][m][x][y] = self.bias[m]
                        for i in range(self.kernel_size[0]):
                            for j in range(self.kernel_size[1]):
                                for k in range(self.in_channels):
                                    self.out[n][m][x][y] += inp[n][k][self.stride[0]*x+i-1][self.stride[1]*y+j]*self.filter[m][k][i][j]
        return self.out
