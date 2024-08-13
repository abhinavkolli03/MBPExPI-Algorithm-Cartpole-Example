import gym
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
from baselines.common.parallel_sampler import ParallelSampler
from baselines.policy.mlp_policy import MlpPolicy
from baselines.policy.cnn_policy import CnnPolicy
from baselines.optimalMis import opt_pomis
from baselines.common.rllab_utils import Rllab2GymWrapper, rllab_env_from_name
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import get_env_type

import re
import os
import time
import logging
import numpy as np

# Global Functions
def global_make_policy(name, ob_space, ac_space, policy, policy_initializer, learnable_variance, variance_init):
    if policy == 'linear':
        hid_size = []
        num_hid_layers = 0
        use_bias = False
    elif policy == 'simple-nn':
        hid_size = [16]
        num_hid_layers = 1
        use_bias = True
    elif policy == 'nn':
        hid_size = [100, 50, 25]
        num_hid_layers = 3
        use_bias = True
    else:
        raise Exception('Unrecognized policy type.')

    if policy == 'linear' or policy == 'nn' or policy == 'simple-nn':
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         hid_size=hid_size, num_hid_layers=num_hid_layers,
                         gaussian_fixed_var=True, use_bias=use_bias, use_critic=False,
                         hidden_W_init=policy_initializer, output_W_init=policy_initializer,
                         learnable_variance=learnable_variance, variance_initializer=variance_init)
    elif policy == 'cnn':
        return CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                         gaussian_fixed_var=True, use_bias=False, use_critic=False,
                         hidden_W_init=policy_initializer, output_W_init=policy_initializer)
    else:
        raise Exception('Unrecognized policy type.')


def global_make_env(env):
    if env.startswith('rllab.'):
        env_name = re.match('rllab.(\S+)', env).group(1)
        env_rllab_class = rllab_env_from_name(env_name)
        env_rllab = env_rllab_class()
        return Rllab2GymWrapper(env_rllab)
    else:
        env_type = get_env_type(env)
        if env_type == 'atari':
            _env = make_atari(env)
            return wrap_deepmind(_env)
        else:
            return gym.make(env)

def train(env, policy, policy_init, n_episodes, horizon, seed, njobs=1, save_weights=0,
          learnable_variance=True, variance_init=1, **alg_args):

    # Environment and Policy Initialization
    env_instance = global_make_env(env)
    policy_initializer = (
        tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        if policy_init == 'xavier'
        else U.normc_initializer(0.0) if policy_init == 'zeros'
        else U.normc_initializer(0.1) if policy_init == 'small-weights'
        else None
    )

    def make_policy(name, ob_space, ac_space):
        return global_make_policy(name, ob_space, ac_space, policy, policy_initializer, learnable_variance, variance_init)

    # sampler = ParallelSampler(make_policy, lambda: env_instance, n_episodes, horizon, True, n_workers=njobs, seed=seed)

    try:
        affinity = len(os.sched_getaffinity(0))
    except:
        affinity = njobs

    sess = U.make_session(affinity)
    sess.__enter__()

    set_global_seeds(seed)

    gym.logger.setLevel(logging.DEBUG)

    opt_pomis.learn(lambda: env_instance, make_policy, n_episodes=n_episodes, horizon=horizon,
                    save_weights=save_weights, learnable_variance=learnable_variance,
                    variance_initializer=variance_init, **alg_args)

    # opt_pomis.learn(lambda: env_instance, make_policy, n_episodes=n_episodes, horizon=horizon,
    #                 sampler=sampler, save_weights=save_weights, learnable_variance=learnable_variance,
    #                 variance_initializer=variance_init, **alg_args)

    # sampler.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--policy', type=str, default='nn')
    parser.add_argument('--policy_init', type=str, default='xavier')
    parser.add_argument('--max_iters', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--save_weights', type=int, default=0)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--iw_method', type=str, default='is')
    parser.add_argument('--iw_norm', type=str, default='none')
    parser.add_argument('--natural', type=bool, default=False)
    parser.add_argument('--bound', type=str, default='max-d2-harmonic')
    parser.add_argument('--delta', type=float, default=0.99)
    parser.add_argument('--njobs', type=int, default=-1)
    parser.add_argument('--center', type=bool, default=False)
    parser.add_argument('--clipping', type=bool, default=False)
    parser.add_argument('--warm_start', type=bool, default=True)
    parser.add_argument('--entropy', type=str, default='none')
    parser.add_argument('--reward_clustering', type=str, default='none')
    parser.add_argument('--experiment_name', type=str, default='none')
    parser.add_argument('--capacity', type=int, default=10)
    parser.add_argument('--inner', type=int, default=10)
    parser.add_argument('--learnable_variance', type=bool, default=False)
    parser.add_argument('--variance_init', type=float, default=-1)
    parser.add_argument('--penalization', type=bool, default=False)
    parser.add_argument('--constant_step_size', type=float, default=0)
    parser.add_argument('--shift_return', type=bool, default=False)
    parser.add_argument('--power', type=float, default=1)
    parser.add_argument('--file_name', type=str, default="")

    args = parser.parse_args()
    if args.file_name == 'progress':
        file_name = '%s_delta=%s_seed=%s_%s' % (args.env.upper(), args.delta, args.seed, time.time())
    else:
        file_name = args.file_name

    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)

    train(env=args.env,
          policy=args.policy,
          policy_init=args.policy_init,
          n_episodes=args.num_episodes,
          horizon=args.horizon,
          seed=args.seed,
          save_weights=args.save_weights,
          max_iters=args.max_iters,
          iw_method=args.iw_method,
          iw_norm=args.iw_norm,
          use_natural_gradient=args.natural,
          bound=args.bound,
          delta=args.delta,
          gamma=args.gamma,
          max_offline_iters=args.max_offline_iters,
          center_return=args.center,
          clipping=args.clipping,
          entropy=args.entropy,
          reward_clustering=args.reward_clustering,
          capacity=args.capacity,
          inner=args.inner,
          warm_start=args.warm_start,
          learnable_variance=args.learnable_variance,
          variance_init=args.variance_init,
          penalization=args.penalization,
          constant_step_size=args.constant_step_size,
          shift_return=args.shift_return,
          power=args.power)

if __name__ == '__main__':
    main()
