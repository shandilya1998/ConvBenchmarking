from test import Test
import conv
from tqdm import tqdm

def run(
    imp, 
    gpu = False
    ):
    layer = None
    if imp == 'naive':
        layer = conv.naive.Conv2d
    elif imp == 'winograd':
        layer = conv.winograd.Conv2d
    elif imp == 'im2col':
        layer = conv.im2col.Conv2d
    elif layer == 'fft':
        layer = conv.fft.Conv2d
    elif layer == 'cp_conv':
        layer = conv.cp_conv.Conv2d
    elif layer == 'tucker_conv':
        layer = conv.tucker_conv.Conv2d
    else:
        raise ValueError('expected one of `naive`, `winograd`, `im2col`, `fft`, `cp_conv` or `tucker_conv`, got `{val}`'.format(val = imp))

    test = Test(layer, gpu)
    batch_size = 128 if gpu else 8
    in_channels = [1, 3, 128, 1024]
    out_channels = [3, 64, 512, 1024]
    kernel_size = [3, 5, 15, 64, 128]
    stride = [1]
    in_size = [1024, 512, 256]
    configs = []
    for c in in_channels:
        for m in out_channels:
            for r in kernel_size:
                for s in stride:
                    for size in in_size:
                        config = {}
                        config['implementation'] = imp
                        config['in_channels'] = c
                        config['out_channels'] = m 
                        config['kernel_size'] = r
                        config['stride'] = s
                        config['size'] = size
                        config['batch_size'] = batch_size
                        configs.append(config)

    results = []
    for config in tqdm(configs):
        test.build(config)
        x = torch.rand(
            (
                config['batch_size'], 
                config['in_channels'],
                config['size'],
                config['size']
            )
        )
        results.append(test(x))

    df = pd.DataFrane(results)

    df.to_csv('outputs/results.csv') 
