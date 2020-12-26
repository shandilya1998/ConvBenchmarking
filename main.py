from run import run
import argparse
import random

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', 
        type=str2bool, 
        nargs='?', 
        const=True, 
        default=False, 
        help='use gpu or not'
    )
    parser.add_argument(
        '--job-dir', 
        help='path for saving images in gcs'
    )
    args = parser.parse_args()

    imps = ['cp_conv', 'naive', 'winograd', 'im2col', 'fft']

    random.shuffle(imps)

    results = []

    device = 'cpu'
    if args.gpu:
        device = 'gpu'

    for imp in imps:
        print(
            'running implementation `{imp}` on device `{dev}`'.format(
                imp = imp,
                dev = device
            )
        )
        results = run(imp, results, args.gpu)
    
    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(
            args.job_dir, 
            'results_{dev}.csv'.format(
                dev = device
            )
        )
    )
