import math
import torch
from . import wincnn

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
            self.kernel_size = kernel_size
        elif isinstance(kernel_size, tuple):
            if kernel_size[0] != kernel_size[1]:
                raise(
                    ValueError(
                        'Expected a square filter, got a filter of shape `{shape}`'.format(shape = kernel_size)
                    )
                )
            self.kernel_size = kernel_size[0]
        else:
            raise ValueError('Kernel size must be an int or a tuple got {t}'.format(t = type(kernel_size)))

        if isinstance(stride, int):
            self.stride = stride
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise ValueError('Kernel size must be an int or a tuple got {t}'.format(t = type(kernel_size)))
        if self.stride != 1:
            raise(ValueError('Only stride of 1 supported, got `{s}`'.format(s = self.stride)))
        self.use_bias = bias
        if bias:
            self.bias = torch.Tensor(self.out_channels)
            self.bias = torch.nn.Parameter(self.bias)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self.filter = torch.Tensor(
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        self.filter = torch.nn.Parameter(self.filter)
        torch.nn.init.kaiming_uniform_(self.filter, a=math.sqrt(5))
        a = []
        for i in range(math.floor(kernel_size/2)+1):
            if i ==0:
                a.append(0)
            else:
                a.extend([i, -i])
        a = tuple(a)
        self.A_T, self.G, self.B_T, self.f = wincnn.cookToomFilter(
            a, 
            2,
            kernel_size
        )
        self.A_T = torch.Tensor(self.A_T.tolist())
        self.G = torch.Tensor(self.G.tolist())
        self.B_T = torch.Tensor(self.B_T.tolist())
        self.f = torch.Tensor(self.f.tolist())
        self.B = self.B_T.transpose(1, 0)
        self.G_T = self.G.transpose(1, 0)
        self.A = self.A_T.transpose(1, 0)

    def calculate_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.out_channels,
            int((input_shape[-2]-self.kernel_size)/self.stride + 1),
            int((input_shape[-1]-self.kernel_size)/self.stride + 1)
        )

    def forward(self, x):
        """
        Compute Winograd convolution.
        :param input:
        :param filter:
        :return: output
        """
        N, C, H, W = x.size()
        assert C == self.in_channels
        assert H == W
        m = 2
        a = m + self.kernel_size - 1
        # TODO pad with zeros the input for perfect tiling and slice the output.
        overlap = self.kernel_size - 1
        if (H >= 4 and H % 2 == 0) is False:
            raise Exception("Only input for perfect tiling is supported.")
        x = torch.transpose(x, 0, 1)
        assert x.size() == (self.in_channels, N, H, W)
        # ntile = int(math.ceil(H//a))
        # P = N * ntile * ntile
        T = (W - a) // overlap + 1  # tiles_per_channel
        P = N * T * T
        U = torch.zeros(self.out_channels, self.in_channels, a, a)
        V = torch.zeros(self.in_channels, P, a, a)
        for k in range(self.out_channels):
            for c in range(self.in_channels):
                U[k, c] = torch.matmul(
                    self.G,
                   torch.matmul(self.filter[k, c], self.G_T)
                )
        for n in range(N):
            for tH in range(T):
                for tW in range(T):
                    for c in range(self.in_channels):
                        b = n * (T * T) + tH * T + tW
                        vH = tH * (self.kernel_size - 1)
                        vW = tW * (self.kernel_size - 1)
                        V[c, b] = torch.matmul(self.B_T, torch.matmul(
                            x[c, n, vH:vH + a, vW:vW + a], self.B))
        M = torch.zeros(self.out_channels, P, a, a)
        for k in range(self.out_channels):
            for b in range(P):
                for c in range(self.in_channels):
                    M[k, b] += U[k, c] * V[c, b]
        # M = torch.matmul(U, V)
        out_size = H - self.kernel_size + 1
        Y = torch.zeros(self.out_channels, N, out_size, out_size)
        for k in range(self.out_channels):
            for n in range(N):
                for tH in range(T):
                    for tW in range(T):
                        b = n * (T * T) + tH * T + tW
                        oH = tH * m
                        oW = tW * m
                        Y[k, n, oH:oH + m, oW:oW + m] = torch.matmul(
                            self.A_T, torch.matmul(M[k, b], self.A))

        Y = torch.transpose(Y, 0, 1)
        return Y
