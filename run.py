from test import Test
from conv import naive, winograd, im2col, cp_conv, fft
from tqdm import tqdm
import torch

def run(
    imp,
    results,
    gpu = False,
    ):
    layer = None
    if imp == 'naive':
        layer = naive.Conv2d
    elif imp == 'winograd':
        layer = winograd.Conv2d
    elif imp == 'im2col':
        layer = im2col.Conv2d
    elif imp == 'fft':
        layer = fft.Conv2d
    elif imp == 'cp_conv':
        layer = cp_conv.Conv2d
    else:
        raise( 
            ValueError(
                'expected one of `naive`, `winograd`, `im2col`, `fft`, `cp_conv` or `tucker_conv`, got `{val}`'.format(
                    val = imp
                )
            )
        )

    test = Test(layer, gpu)
    batch_size = 128 if gpu else 8
    in_channels = [3, 64, 512]
    out_channels = [3, 64, 512]
    kernel_size = [3, 15, 64]
    stride = [1]
    in_size = [128, 1024]
    configs = []
 
    for c in in_channels:
        for m in out_channels:
            for size in in_size:
                for s in stride:
                    for r in kernel_size:                        
                        config = {}
                        config['implementation'] = imp
                        config['in_channels'] = c
                        config['out_channels'] = m 
                        config['kernel_size'] = r
                        config['stride'] = s
                        config['size'] = size
                        config['batch_size'] = batch_size
                        config['gpu'] = gpu
                        if config['in_channels'] != config['out_channels']:
                            configs.append(config)

    print('number of configurations')
    print(len(configs))
     
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
        if gpu:
            x = x.cuda() 
        results.append(test(x))

    return results 
