from conv import *
import time

class Test:
    def __init__(self, imp, gpu):
        self.imp = imp
        self.gpu = gpu

    def build(self, config):
        self.config = config
        self.layer = self.imp(
            in_channels = config['in_channels'],
            out_channels = config['out_channels'],
            kernel_size = config['kernel_size'],
            stride = config['stride'],
        )
        if config['gpu']:
            self.layer = self.layer.cuda()
        self.built = True

    def reset(self):
        self.built = False

    def __call__(self, x):
        if not self.built:
            raise AttributeError('Test not built, call `build()` to build and run test')
        start = time.time()
        out = self.layer(x)
        end = time.time()
        latency = end-start
        self.config['latency'] = latency
        self.config['input_shape'] = x.shape
        self.config['output_shape'] = out.shape
        return self.config
