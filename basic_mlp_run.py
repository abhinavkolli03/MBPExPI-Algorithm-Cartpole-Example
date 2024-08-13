import gym
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.common import set_global_seeds
from baselines import logger
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

def train(env, policy, policy_init, n_episodes, horizon, seed, save_weights=0,
          learnable_variance=True, variance_init=1, use_opt_pomis=False, **alg_args):

    # Environment and Policy Initialization
    env_instance = global_make_env(env)
    policy_initializer = (
        tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        if policy_init == 'xavier'
        else U.normc_initializer(0.0) if policy_init == 'zeros'
        else U.normc_initializer(0.1) if policy_init == 'small-weights'
        else None
    )

    sess = U.single_threaded_session()
    sess.__enter__()
    with sess.as_default():
        set_global_seeds(seed)

        # Create policy
        policy_instance = global_make_policy('policy', env_instance.observation_space, env_instance.action_space, 
                                             policy, policy_initializer, learnable_variance, variance_init)
        
        U.initialize()
        gym.logger.setLevel(logging.DEBUG)

        # if use_opt_pomis:
        #     # Use opt_pomis optimization
        #     opt_pomis.learn(
        #         make_env=lambda: env_instance, 
        #         make_policy=lambda name, ob_space, ac_space: policy_instance,
        #         n_episodes=n_episodes,
        #         horizon=horizon,
        #         save_weights=save_weights,
        #         learnable_variance=learnable_variance,
        #         variance_initializer=variance_init,
        #         **alg_args
        #     )
        # else:
        # Standard training loop
        total_rewards = []
        total_steps = []

        for episode in range(n_episodes):
            ob = env_instance.reset()
            total_reward = 0
            steps = 0
            final_state = None

            while True:
                action, _ = policy_instance.act(True, ob)
                ob, reward, done, _ = env_instance.step(action)
                total_reward += reward
                steps += 1
                final_state = ob
                if done:
                    break

            total_rewards.append(total_reward)
            total_steps.append(steps)

            # Logging
            logger.logkv('TotalReward', total_reward)
            logger.logkv('Episode', episode + 1)
            logger.logkv('Steps', steps)
            logger.dumpkvs()

        # Log the average reward and steps at the end
        avg_reward = sum(total_rewards) / n_episodes
        avg_steps = sum(total_steps) / n_episodes

        logger.logkv('AvgReward', avg_reward)
        logger.logkv('AvgSteps', avg_steps)
        logger.dumpkvs()

        # Save weights if requested
        if save_weights:
            policy_instance.save(f"policy_weights_{env}_{policy}_{seed}.pkl")

    env_instance.close()
    sess.close()

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
    parser.add_argument('--use_opt_pomis', action='store_true', help='Use opt_pomis for training')

    args = parser.parse_args()

    file_name = f'{args.env.upper()}_delta={args.gamma}_seed={args.seed}_{time.time()}'
    logger.configure(dir=args.logdir, format_strs=['stdout', 'csv', 'tensorboard'], file_name=file_name)

    train(env=args.env,
          policy=args.policy,
          policy_init=args.policy_init,
          n_episodes=args.num_episodes,
          horizon=args.horizon,
          seed=args.seed,
          save_weights=args.save_weights,
          max_iters=args.max_iters,
          gamma=args.gamma,
          use_opt_pomis=args.use_opt_pomis)

if __name__ == '__main__':
    main()
