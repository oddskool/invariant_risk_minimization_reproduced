import argparse
import random
from routines import *

parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", '-lr', type=float, default=1e-4)
parser.add_argument("--nb-units", '-n', type=int, default=32)
parser.add_argument("--batch-size", '-bs', type=int, default=128)
parser.add_argument("--epochs", '-e', type=int, default=32)
parser.add_argument("--repeats", '-r', type=int, default=5)
parser.add_argument("--output-filename", '-f', type=str)
parser.add_argument("--warmup-epochs", '-w', type=int, default=1)
parser.add_argument("--lambda-multiplier", '-l', type=float, default=10000)
parser.add_argument("--verbose", '-v', default=False, action='store_true')

def get_acc(args):
    
    def lambda_scheduler(epoch):
        if epoch < args.warmup_epochs:
            return 0
        return args.lambda_multiplier
    
    def run_once():
        irm = IRMModel(
            model = get_model(n_final_units=args.nb_units),
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        )
        irm.train(args.epochs, lambda_scheduler, batch_size=args.batch_size, print_=args.verbose)
        acc = irm.logs['ood-acc'][-1]
        return acc
    
    accs = [run_once() for _ in range(args.repeats)]
    return accs

def run(args):
    print('conf:', args)
    accs = get_acc(args)
    with open(args.output_filename, 'w') as f:
        f.write(str(vars(args))+'\n')
        f.write('%.5f\n'%np.mean(accs))
        f.write(' '.join([str(_) for _ in accs])+'\n')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.output_filename is None:
        args.output_filename = '_'.join(['%s=%s'%(k,v) for k,v in sorted(vars(args).items())])+'.txt'
    run(args)
