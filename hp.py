import random

params = {
 'learning_rate': [1e-3,1e-4,1e-5,1e-6],
 'nb_units': [32],
 'batch_size':  [128,256],
 'epochs': [30,100,500],
 'warmup_epochs': [1, ],
 'lambda_multiplier': [100000, 150000],
}

confs = []
for lr in params['learning_rate']:
    for n in params['nb_units']:
        for b in params['batch_size']:
            for e in params['epochs']:
                for w in params['warmup_epochs']:
                    for l in params['lambda_multiplier']:
                        confs += [{
                            'learning_rate': lr, 
                            'nb_units': n, 
                            'batch_size': b, 
                            'epochs': e, 
                            'warmup_epochs': w, 
                            'lambda_multiplier': l, 
                        }]
                       
print('found', len(confs), 'configs')

for i in range(30):
    c = confs[random.randint(0,len(confs)-1)]
    print(c)
    with open('scripts/run_%d.sh' % i, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('PYTHONPATH=.. python ../run_irm.py -lr {learning_rate} -n {nb_units} -b {batch_size} -e {epochs} -w {warmup_epochs} -l {lambda_multiplier}\n'.format(**c))
