import argparse

from routines import *

parser = argparse.ArgumentParser()
parser.add_argument("--learning-rate", '-lr', type=float, default=1e-4)
parser.add_argument("--nb-units", '-n', type=int, default=256)
parser.add_argument("--nb-layers", '-nl', type=int, default=3)
#parser.add_argument("--batch-size", '-bs', type=int, default=128)
parser.add_argument("--epochs", '-e', type=int, default=32)
parser.add_argument("--repeats", '-r', type=int, default=5)
parser.add_argument("--output-filename", '-f', type=str)
parser.add_argument("--warmup-epochs", '-w', type=int, default=1)
parser.add_argument("--lambda-multiplier", '-l', type=int, default=10000)
parser.add_argument("--l2-reg", '-l2', type=float, default=0)
parser.add_argument("--dropout", '-d', type=float, default=0)
parser.add_argument("--mlp", '-m', default=False, action='store_true')
parser.add_argument("--verbose", '-v', default=False, action='store_true')
parser.add_argument("--eval-every", '-ee', default=10, type=int)
parser.add_argument('--hp-tuning', '-hp', default=0, type=int)


def random_parameters_tuning(args):

    def sample_config(cfg):
        params = dict()
        for k in cfg.keys():
            v = np.random.choice(cfg[k])
            params[k] = v
        return params

    params = {
       'learning_rate': [1e-3, 1e-4, 1e-5, 1e-6],
        'nb_units': [64, 128, 256, 512],
        'nb_layers': [2, 3, 5, 7],
        'mlp': [True, True, True, True, False],
        'epochs': [50, 100, 200, 500, 700],
        'warmup_epochs': [10, 50, 100, 200],
        'lambda_multiplier': [100, 10**3, 10**4, 10**5, 10**6],
    }

    envs = ColoredMNISTEnvironments()

    for i in range(args.hp_tuning):
        cfg = sample_config(params)
        for k, v in cfg.items():
            args.__dict__[k] = v
        print("sampling config #", i+1, '/', args.hp_tuning)
        run(args, override_output_fn=True, envs=envs)


def get_results(args, envs=None):
    
    def lambda_scheduler(epoch):
        if epoch < args.warmup_epochs:
            return 1
        return args.lambda_multiplier #* epoch**4
    
    def run_once(seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        irm = IRMModel(
            model=get_model(args),
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            # optimizer=tf.keras.optimizers.SGD(learning_rate=args.learning_rate),
            envs=envs
        )
        irm.train(args.epochs, lambda_scheduler,
                  print_=args.verbose, eval_every=args.eval_every)
        acc = irm.logs['e33_acc'][-1]
        return acc, irm.logdir
    
    results = [run_once(42*_) for _ in range(args.repeats)]
    return [r[0] for r in results], [r[1] for r in results]


def run(args, override_output_fn=False, envs=None):
    print('conf:', args)
    accuracies, logdirs = get_results(args, envs)
    if args.output_filename is None or override_output_fn:
        args.output_filename = 'results/' + '_'.join(['%s=%s' % (k, v) for k, v in sorted(vars(args).items()) if k != 'output_filename'])+'.txt'
    with open(args.output_filename, 'w') as f:
        f.write(str(vars(args))+'\n')
        f.write('%.5f\n' % np.mean(accuracies))
        f.write(' '.join([str(_) for _ in accuracies])+'\n')
        f.write(' '.join([str(_) for _ in logdirs])+'\n')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.hp_tuning > 0:
        random_parameters_tuning(args)
    else:
        run(args)
