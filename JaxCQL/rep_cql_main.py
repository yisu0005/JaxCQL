import os
import time
from copy import deepcopy
import uuid

import numpy as np
import pprint

import jax
import jax.numpy as jnp
import flax

import gym
import d4rl

import absl.app
import absl.flags

from .rep_cql import REPCQL
from .rep import REP
from .bc_policy import BC
from .replay_buffer import get_d4rl_dataset, subsample_batch
from .jax_utils import batch_to_jax
from .model import TanhGaussianPolicy, FullyConnectedQFunction, SamplerPolicy, SamplerDecoder, Discriminator, ActionDecoder, ActionRepresentationPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=42,
    save_model=False,
    batch_size=256,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,

    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    
    train_decorrelation=True,
    train_bc=True,
    train_cql=True,
    encoder_no_tanh=True,
    action_scale=1.0,
    discriminator_arch='512-256',
    encoder_arch='256-256',
    decoder_arch='256-256',
    decorrelation_method='GAN',
    decorrelation_epochs=200,
    decor_n_train_step_per_epoch=500,
    policy_n_epochs=100,
    policy_n_train_step_per_epoch=500,
    latent_dim=2.0,
    dis_dropout=False,

    rep=REP.get_default_config(),
    bc=BC.get_default_config(),
    cql=REPCQL.get_default_config(),
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

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)

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

    viskit_metrics = {}
    rep = REP(FLAGS.rep, encoder, discriminator, decoder, method=FLAGS.decorrelation_method)
    for epoch in range(FLAGS.decorrelation_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.decor_n_train_step_per_epoch):
                batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                metrics.update(prefix_metrics(rep.train(batch), 'decorrelation'))
        
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

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
                batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                metrics.update(prefix_metrics(bc_agent.train(batch), 'bc'))

        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        
    """
    RL Training (SAC)
    """
    policy = TanhGaussianPolicy(
        observation_dim, latent_action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset, FLAGS.action_scale
    )

    qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    repcql = REPCQL(FLAGS.cql, policy, qf, rep, bc_agent)
    sampler_policy = SamplerPolicy(repcql.policy, repcql.train_params['policy'])
    sampler_decoder = SamplerDecoder(rep.decoder, rep.train_params['decoder'])

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {'epoch': epoch}

        with Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                metrics.update(prefix_metrics(repcql.train(batch, bc=epoch < FLAGS.bc_epochs), 'repcql'))

        with Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.repa_sample(
                    sampler_policy.update_params(repcql.train_params['policy']),
                    sampler_decoder, 
                    FLAGS.eval_n_trajs, deterministic=True
                )

                metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
                metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
                metrics['average_normalizd_return'] = np.mean(
                    [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
                )
                if FLAGS.save_model:
                    save_data = {'repcql': repcql, 'variant': variant, 'epoch': epoch}
                    wandb_logger.save_pickle(save_data, 'model.pkl')

        metrics['train_time'] = train_timer()
        metrics['eval_time'] = eval_timer()
        metrics['epoch_time'] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {'repcql': repcql, 'variant': variant, 'epoch': epoch}
        wandb_logger.save_pickle(save_data, 'model.pkl')

if __name__ == '__main__':
    absl.app.run(main)
