from action_noise import *
from param_noise import *

def hyperparam_search():
    print("Starting hyperparam-optimization")

    params = {'tau': hp.uniform('tau', 0.001, 0.01),
              'critic_lr': hp.uniform('critic_lr', 0.0002, 0.02),
              'actor_lr': hp.uniform('actor_lr', 0.0001, 0.01),
              'gamma': hp.uniform('gamma', 0.9, 0.999),
              'noise_std_dev': hp.uniform('noise_std_dev', 0.1, 0.4), }

    best = fmin(train, params,
                algo=hyperopt.rand.suggest, max_evals=100)

    print(best)
    print(hyperopt.space_eval(params, best))



def train()
