import torch

class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        activation,
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
        self.out = torch.zeros(bs, self.out_channels, h_out, w_out)  
        self.activation = activation

    def calculate_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.out_channels,
            int((input_shape[-2]-self.kernel_size[0])/self.stride[0] + 1),
            int((input_shape[-1]-self.kernel_size[1])/self.stride[1] + 1)
        )

    def im2col(self, x, bs, h, w):
        rows = []
        for b in range(bs):
            for i in range(0, h-self.kernel_size[0]+1, self.stride[0]):
                for j in range(0, w-self.kernel_size[1])+1, self.stride[1]:
                    inp = x[
                        b, 
                        :, 
                        i:i+self.kernel_size[0], 
                        j:j+self.kernel_size[1]
                    ]
                    rows.append(inp.flatten().unsqueeze(0))
        rows = torch.cat(rows, dim = 0).transpose(0, 1)
        return rows
        
    def col2im(self, x, bs, h_out, w_out):
        cols = x.shape[-1]
        items = []
        for i in range(0, cols, h_out*w_out):
            item = x[:, col:col+h_out*w_out]
            items.append(item.reshape(self.out_channels, h_out, w_out).unsqueeze(0))
        out = torch.cat(items, dim = 0)
        return out
                      

    def forward(self, x):
        f = self.filter.flatten(1, -1)
        bs, _, h_out, w_out = self.calculate_output_shape(x.shape)
        h = x.shape[2]
        w = x.shape[3]
        x_rows = self.im2col(x, bs, h, w)
        out_rows = torch.matmul(f, x_rows)
        out = self.col2im(out_rows, bs, h_out, w_out)
        return out
