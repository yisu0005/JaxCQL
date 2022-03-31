import os
import time
from copy import deepcopy
import uuid
import numpy as np
import pprint
import cloudpickle as pickle

import jax
import jax.numpy as jnp
import flax

import gym
import d4rl

import absl.app
import absl.flags

from .de_cql import DECQL
from .bc_policy import BC
from .replay_buffer import get_d4rl_dataset, subsample_batch, get_top_dataset
from .jax_utils import batch_to_jax, next_rng
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy, SamplerDecoder, SamplerEncoder, Discriminator, ActionDecoder, ActionRepresentationPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    rep_seed=42,
    save_model=False,
    batch_size=256,
    rep_batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,
    policy_entropy_scale=2.0, 

    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=20,
    eval_n_trajs=10,
    
    train_bc=True,
    train_cql=True,
    encoder_no_tanh=True,
    action_scale=1.0,
    discriminator_arch='512-256-128-64',
    encoder_arch='256-256-256',
    decoder_arch='256-256-256',
    decorrelation_top_data=False,
    decorrelation_method='GAN',
    decorrelation_epochs=40, #500
    decor_n_train_step_per_epoch=1000,
    policy_n_epochs=10,
    policy_n_train_step_per_epoch=500,
    latent_dim=2.0,
    dis_dropout=False,
    bc_filter_success=True,
    bc_filter_percentile=100.0,

    bc=BC.get_default_config(),
    cql=DECQL.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
    FLAGS = absl.flags.FLAGS

    variant = get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False
    )

    model_dir = 'antmaze_models'
    os.makedirs(model_dir, exist_ok=True)

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

    if FLAGS.bc_filter_success:
        top_dataset = get_top_dataset(dataset)
    else:
        top_dataset = get_top_dataset(dataset, filter_success=False, percentile=FLAGS.bc_filter_percentile)

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]
    latent_action_dim = int(FLAGS.latent_dim * action_dim)

    """
    Decorrelation Training
    """

    encoder = ActionRepresentationPolicy(
        observation_dim,
        action_dim, 
        latent_action_dim, 
        FLAGS.encoder_arch,
        FLAGS.orthogonal_init,
        FLAGS.encoder_no_tanh,
        FLAGS.policy_log_std_multiplier,
        FLAGS.policy_log_std_offset,
        # FLAGS.action_scale,
        )

    discriminator = Discriminator(
        observation_dim, 
        latent_action_dim,
        FLAGS.discriminator_arch,
        FLAGS.dis_dropout,
    )

    decoder = ActionDecoder(
       observation_dim,
       latent_action_dim,
       action_dim,
       FLAGS.decoder_arch, 
       FLAGS.orthogonal_init, 
    )
   
    if FLAGS.train_bc:
        """
        BC Training
        """
        logger_policy = TanhGaussianPolicy(
            observation_dim, 
            action_dim, 
            FLAGS.policy_arch,
            FLAGS.orthogonal_init,
            FLAGS.policy_log_std_multiplier,
            FLAGS.policy_log_std_offset,
        )

        bc_agent = BC(FLAGS.bc, logger_policy)

        viskit_metrics = {}
        for epoch in range(FLAGS.policy_n_epochs):
            metrics = {'epoch': epoch}

            with Timer() as train_timer:
                for batch_idx in range(FLAGS.policy_n_train_step_per_epoch):
                    batch = batch_to_jax(subsample_batch(top_dataset, FLAGS.batch_size))
                    metrics.update(prefix_metrics(bc_agent.train(batch), 'bc'))

            wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)


    if FLAGS.train_cql:
        
        """
        RL Training (SAC)
        """
        policy = TanhGaussianPolicy(
            observation_dim, latent_action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
            FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset, FLAGS.action_scale
        )

        qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

        if FLAGS.cql.target_entropy >= 0.0:
            FLAGS.cql.target_entropy = -(np.prod(eval_sampler.env.action_space.shape) * FLAGS.policy_entropy_scale).item()
        

        decql = DECQL(FLAGS.cql, policy, qf, encoder, decoder, discriminator, bc_agent)
        sampler_policy = SamplerPolicy(decql.policy, decql.train_params['policy'])
        sampler_decoder = SamplerDecoder(decql.decoder, decql.train_params['decoder'])

        viskit_metrics = {}
        for epoch in range(FLAGS.n_epochs):
            metrics = {'epoch': epoch}

            with Timer() as train_timer:
                for batch_idx in range(FLAGS.n_train_step_per_epoch):
                    batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                    metrics.update(prefix_metrics(decql.train(batch, bc=epoch < FLAGS.bc_epochs, decorrelation=epoch < FLAGS.decorrelation_epochs), 'decql'))

            with Timer() as eval_timer:
                if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                    trajs = eval_sampler.repa_sample(
                        sampler_policy.update_params(decql.train_params['policy']),
                        sampler_decoder.update_params(decql.train_params['decoder']), 
                        FLAGS.eval_n_trajs, deterministic=True
                    )

                    metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                    metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                    metrics['average_normalizd_return'] = np.mean(
                        [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                    )
                    if FLAGS.save_model:
                        save_data = {'decql': decql, 'variant': variant, 'epoch': epoch}
                        wandb_logger.save_pickle(save_data, 'model.pkl')

            metrics['train_time'] = train_timer()
            metrics['eval_time'] = eval_timer()
            metrics['epoch_time'] = train_timer() + eval_timer()
            wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        if FLAGS.save_model:
            save_data = {'decql': decql, 'variant': variant, 'epoch': epoch}
            wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)