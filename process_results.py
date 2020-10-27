from glob import glob

results = []
for fn in glob('results/*.txt'):
    cfg, mean_ood_acc, ood_accuracies, logdirs = open(fn).read().strip().split('\n')
    results += [(float(mean_ood_acc), eval(cfg))]

params = [
    'learning_rate',
    'nb_units',
    'nb_layers',
    'mlp',
    'epochs',
    'warmup_epochs',
    'lambda_multiplier',
]

print('\n'.join(["%s %.3f %s" % (
    "[*]" if _[0] > .7 else "[ ]",
    _[0],
    ' '.join(sorted(["%s:%s" % (k, v) for k, v in _[1].items() if k in params]))
) for _ in sorted(results, reverse=True)]))
