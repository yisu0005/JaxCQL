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

from .rep_cql import REPCQL
from .rep import REP
from .bc_policy import BC
from .replay_buffer import get_d4rl_dataset, get_preprocessed_dataset, subsample_batch, get_top_dataset, get_sarsa_dataset, get_preprocessed_dataset
from .jax_utils import batch_to_jax, next_rng
from .model import TanhGaussianPolicy, FullyConnectedQFunction, FullyConnectedQFunction, FullyConnectedActionQFunction, SamplerPolicy, SamplerDecoder, SamplerEncoder, Discriminator, ActionDecoder, ActionSeperatedDecoder, ActionRepresentationPolicy
from .sampler import StepSampler, TrajSampler
from .utils import Timer, define_flags_with_default, set_random_seed, print_flags, get_user_flags, prefix_metrics
from .utils import WandBLogger, random_split
from viskit.logging import logger, setup_logger


FLAGS_DEF = define_flags_with_default(
    env='halfcheetah-medium-v2',
    max_traj_length=1000,
    seed=1,
    rep_seed=42,
    save_model=False,
    batch_size=256,
    rep_batch_size=512,

    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,

    policy_arch='256-256',
    qf_arch='256-256',
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,
    policy_entropy_scale=2.0, 
    qf_normalize=False,
    qf_seperate_action=True,

    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=20,
    eval_n_trajs=10,
    
    train_decorrelation=True,
    train_bc=True,
    train_cql=True,
    encoder_no_tanh=True,
    action_seperate_decoder=True,
    action_scale=1.0,
    discriminator_arch='256-256-256',
    encoder_arch='256-256-256-256',
    decoder_arch='256-256-256',
    decorrelation_method='GAN',
    decorrelation_epochs=500, 
    decor_n_train_step_per_epoch=1000,
    policy_n_epochs=100,
    policy_n_train_step_per_epoch=500,
    latent_dim=2.0,
    dis_dropout=False,
    bc_filter_success=True,
    bc_filter_percentile=100.0,

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

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]
    latent_action_dim = int(FLAGS.latent_dim * action_dim)
    

    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset['rewards'] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action)


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

    action_discriminator = Discriminator(
        observation_dim, 
        action_dim,
        FLAGS.discriminator_arch,
        FLAGS.dis_dropout,
    )

    if FLAGS.action_seperate_decoder:
        decoder = ActionSeperatedDecoder(
        observation_dim,
        latent_action_dim,
        action_dim,
        FLAGS.decoder_arch, 
        FLAGS.orthogonal_init, 
        )
    else:
        decoder = ActionDecoder(
        observation_dim,
        latent_action_dim,
        action_dim,
        FLAGS.decoder_arch, 
        FLAGS.orthogonal_init, 
        )



    rep = REP(FLAGS.rep, encoder, discriminator, decoder, action_discriminator, method=FLAGS.decorrelation_method)

    if FLAGS.train_decorrelation:
        viskit_metrics = {}
        for epoch in range(FLAGS.decorrelation_epochs):
            metrics = {'epoch': epoch}

            with Timer() as train_timer:
                for batch_idx in range(FLAGS.decor_n_train_step_per_epoch):
                    batch = batch_to_jax(subsample_batch(dataset, FLAGS.rep_batch_size))
                    metrics.update(prefix_metrics(rep.train(batch), 'decorrelation'))
            
            # with Timer() as eval_timer:
            #     if epoch == 0 or (epoch + 1) % 10 == 0:
            #         for batch_idx in range(FLAGS.decor_n_train_step_per_epoch):
            #             batch = batch_to_jax(subsample_batch(val_dataset, FLAGS.rep_batch_size))
            #             metrics.update(prefix_metrics(rep.val(batch), 'decorrelation_validation'))
            
            wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        filename = '-'.join([FLAGS.env, str(FLAGS.seed), str(rep.config.recon_alpha), str(rep.config.z_alpha), str(FLAGS.rep_batch_size), str(FLAGS.decorrelation_epochs)]) + '.pkl' 

        with open(os.path.join(model_dir, filename), 'wb') as fout:
            rep_data = {'rep': rep, 'variant': variant, 'epoch': epoch}
            pickle.dump(rep_data, fout)
    else:
        filename = '-'.join([FLAGS.env, str(FLAGS.rep_seed), str(rep.config.recon_alpha), str(rep.config.z_alpha), str(FLAGS.rep_batch_size), str(FLAGS.decorrelation_epochs)]) + '.pkl' 

        with open(os.path.join(model_dir, filename), 'rb') as fin:
            rep_data = pickle.load(fin)
            rep = rep_data['rep']

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
                    batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                    metrics.update(prefix_metrics(bc_agent.train(batch), 'bc'))

            wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # """
    # Disganosis for U(-1,1)
    # """
    # sampler_encoder = SamplerEncoder(rep.encoder, rep.train_params['encoder']) 
    # sampler_decoder = SamplerDecoder(rep.decoder, rep.train_params['decoder'])
    # trajs = eval_sampler.repa_sample(
    #                     None,
    #                     sampler_decoder, 
    #                     FLAGS.eval_n_trajs, deterministic=True,
    #                     latent_ac_dim=latent_action_dim,
    #                 )

    # decoder_metrics = {}
    # decoder_metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
    # decoder_metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
    # decoder_metrics['average_normalizd_return'] = np.mean(
    #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
    # )
    # print(decoder_metrics)

    # """
    # BC
    # """
    # bc_sampler_policy = SamplerPolicy(bc_agent.policy, bc_agent.train_params['policy'])
    # trajs = eval_sampler.sample(
    #                     bc_sampler_policy,
    #                     FLAGS.eval_n_trajs, deterministic=True,
    #                 )

    # decoder_metrics = {}
    # decoder_metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
    # decoder_metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
    # decoder_metrics['average_normalizd_return'] = np.mean(
    #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
    # )
    # print(decoder_metrics)

    # """
    # Encoded/Decoded BC
    # """
    # trajs = eval_sampler.sample_decoded(
    #     next_rng(), bc_sampler_policy, sampler_encoder, sampler_decoder, FLAGS.eval_n_trajs, deterministic=True,
    # )
    # decoder_metrics = {}
    # decoder_metrics['average_return'] = np.mean([np.sum(t['rewards']) for t in trajs])
    # decoder_metrics['average_traj_length'] = np.mean([len(t['rewards']) for t in trajs])
    # decoder_metrics['average_normalizd_return'] = np.mean(
    #     [eval_sampler.env.get_normalized_score(np.sum(t['rewards'])) for t in trajs]
    # )
    # print(decoder_metrics)



    if FLAGS.train_cql:
        
        """
        RL Training (SAC)
        """
        policy = TanhGaussianPolicy(
            observation_dim, latent_action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
            FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset, FLAGS.action_scale
        )

        if FLAGS.qf_seperate_action:
            qf = FullyConnectedActionQFunction(observation_dim, action_dim, 1, FLAGS.qf_arch, FLAGS.orthogonal_init, FLAGS.qf_normalize)
        else:
            qf = FullyConnectedQFunction(observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)


        if FLAGS.cql.target_entropy >= 0.0:
            FLAGS.cql.target_entropy = -(np.prod(eval_sampler.env.action_space.shape) * FLAGS.policy_entropy_scale).item()
        

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
