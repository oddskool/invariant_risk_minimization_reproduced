import argparse
import random
from routines import *

parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", '-lr', type=float, default=1e-4)
parser.add_argument("--nb-units", '-n', type=int, default=32)
parser.add_argument("--nb-layers", '-nl', type=int, default=3)
parser.add_argument("--batch-size", '-bs', type=int, default=128)
parser.add_argument("--epochs", '-e', type=int, default=32)
parser.add_argument("--repeats", '-r', type=int, default=5)
parser.add_argument("--output-filename", '-f', type=str)
parser.add_argument("--warmup-epochs", '-w', type=int, default=1)
parser.add_argument("--lambda-multiplier", '-l', type=float, default=10000)
parser.add_argument("--l2-reg", '-l2', type=float, default=0)
parser.add_argument("--mlp", '-m', default=False, action='store_true')
parser.add_argument("--verbose", '-v', default=False, action='store_true')

def get_results(args):
    
    def lambda_scheduler(epoch):
        if epoch < args.warmup_epochs:
            return 1.
        return args.lambda_multiplier
    
    def run_once():
        irm = IRMModel(
            model = get_model(args),
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        )
        irm.train(args.epochs, lambda_scheduler, batch_size=args.batch_size, print_=args.verbose)
        acc = irm.logs['ood-acc'][-1]
        return acc, irm.logdir
    
    results = [run_once() for _ in range(args.repeats)]
    return [r[0] for r in results], [r[1] for r in results]

def run(args):
    print('conf:', args)
    accuracies, logdirs = get_results(args)
    with open(args.output_filename, 'w') as f:
        f.write(str(vars(args))+'\n')
        f.write('%.5f\n'%np.mean(accuracies))
        f.write(' '.join([str(_) for _ in accuracies])+'\n')
        f.write(' '.join([str(_) for _ in logdirs])+'\n')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.output_filename is None:
        args.output_filename = '_'.join(['%s=%s'%(k,v) for k,v in sorted(vars(args).items())])+'.txt'
    run(args)
