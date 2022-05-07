from binascii import rledecode_hqx
from collections import OrderedDict
from copy import deepcopy
from functools import partial

from ml_collections import ConfigDict

import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax

from .jax_utils import next_rng, value_and_multi_grad, mse_loss
from .model import Scalar, update_target_network
from .utils import prefix_metrics


class REPCQL(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.reg_type = 'none'
        config.soft_target_update_rate = 5e-3
        config.use_cql = True
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 0.0
        config.cql_max_target_backup = True 
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf
        config.epsilon = 1e-4
        config.distance_logging = True
        config.q_value_clip_min = -1000.0
        config.q_value_clip_max = 1000.0
        config.deterministic_action= False

        config.encoder_lr = 3e-4
        config.dis_lr = 3e-4
        config.decoder_lr = 3e-4
        config.optimizer_type = 'adam'
        config.decoder_optimizer_type = 'adam'
        config.encoder_optimizer_type = 'adam'
        config.optimizer_b1 = 0.5 
        config.optimizer_b2 = 0.999
        config.recon_alpha = 0.01
        config.prior = 'uniform'
        config.smooth_dis = False
        config.smooth_decoder = False
        config.latent_stats_logging = True



        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf, encoder, discriminator, decoder, bc_agent):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.observation_dim = policy.observation_dim
        self.action_dim = decoder.action_dim
        self.latent_action_dim = decoder.latent_action_dim
        self.bc_policy = bc_agent.policy
        self.bc_agent = bc_agent
        self.latent_scale = policy.action_scale
        self.dropout = discriminator.dropout
        

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]


        discriminator_params = self.discriminator.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.latent_action_dim)))
        self._train_states['discriminator'] = TrainState.create(
            params=discriminator_params,
            tx=optimizer_class(self.config.dis_lr, b1=self.config.optimizer_b1, b2=self.config.optimizer_b2),
            apply_fn=None,
        )

        decoder_params = self.decoder.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.latent_action_dim)))
        self._train_states['decoder'] = TrainState.create(
            params=decoder_params,
            tx=optimizer_class(self.config.decoder_lr, b1=self.config.optimizer_b1, b2=self.config.optimizer_b2),
            apply_fn=None,
        )

        encoder_params = self.encoder.init(next_rng(), next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states['encoder'] = TrainState.create(
            params=encoder_params,
            tx=optimizer_class(self.config.encoder_lr, b1=self.config.optimizer_b1, b2=self.config.optimizer_b2),
            apply_fn=None,
        )

        policy_params = self.policy.init(next_rng(), next_rng(), jnp.zeros((10, self.observation_dim)))
        self._train_states['policy'] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None
        )

        qf1_params = self.qf.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states['qf1'] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        qf2_params = self.qf.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states['qf2'] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )

        self._target_qf_params = deepcopy({'qf1': qf1_params, 'qf2': qf2_params})

        model_keys = ['policy', 'qf1', 'qf2', 'encoder', 'decoder', 'discriminator']

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states['log_alpha'] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha')

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self._train_states['log_alpha_prime'] = TrainState.create(
                params=self.log_alpha_prime.init(next_rng()),
                tx=optimizer_class(self.config.qf_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha_prime')

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch, rl=True):
        self._total_steps += 1
        self._train_states, self._target_qf_params, metrics = self._train_step(
            self._train_states, self._target_qf_params, next_rng(), batch, rl,
        )
        return metrics

    @partial(jax.jit, static_argnames=('self', 'rl'))
    def _train_step(self, train_states, target_qf_params, rng, batch, rl=True):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            actions = batch['actions']
            rewards = batch['rewards']
            next_observations = batch['next_observations']
            dones = batch['dones']

            loss_collection = {}

            """ GAN Training"""
            batch_size, _ = jnp.shape(observations)
            adversarial_loss = mse_loss

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            (latent_actions, _) = self.encoder.apply(train_params['encoder'], split_rng, observations, actions)

            if self.config.prior == 'uniform':
                rng, split_rng = jax.random.split(rng)
                marginals = jax.random.uniform(split_rng, (batch_size, self.latent_action_dim), minval=-1.0, maxval=1.0)
            elif self.config.prior == 'gaussian':
                rng, split_rng = jax.random.split(rng)
                marginals = jax.random.multivariate_normal(split_rng, jnp.zeros(self.latent_action_dim), jnp.diag(jnp.ones(self.latent_action_dim)), (batch_size, ))
                
            valid = jnp.ones((batch_size))
            fake = jnp.zeros((batch_size))

            g_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, latent_actions), valid)
            decoded_actions = self.decoder.apply(train_params['decoder'], observations, latent_actions)
            reconstruct_loss = mse_loss(decoded_actions, actions)

            rep_loss = g_loss + self.config.recon_alpha * reconstruct_loss
            encoder_loss = rep_loss
            loss_collection['encoder'] = encoder_loss 


            decoder_loss = reconstruct_loss 
            loss_collection['decoder'] = decoder_loss

            if self.config.smooth_dis:
                rng, split_rng = jax.random.split(rng)
                noise = jax.random.uniform(split_rng, jnp.shape(valid), minval=-1.0, maxval=1.0) * 0.2
                valid = valid + noise

                rng, split_rng = jax.random.split(rng)
                noise = jax.random.uniform(split_rng, jnp.shape(valid), minval=0.0, maxval=1.0) * 0.2
                fake = fake + noise 

            if self.dropout:
                rng, split_rng, split_rng2 = jax.random.split(rng, 3) 
                real_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, marginals, train=True,
                            rngs={'dropout': split_rng}), valid)
                latent_actions_cp = jax.lax.stop_gradient(latent_actions)
                fake_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, latent_actions_cp, train=True,
                            rngs={'dropout': split_rng2}), fake)
            else:
                real_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, marginals), valid)
                fake_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, latent_actions), fake)

            d_loss = (real_loss + fake_loss) / 2
            loss_collection['discriminator'] = d_loss

            ### Accuracy ###
            real_result = jax.lax.stop_gradient(self.discriminator.apply(train_params['discriminator'], observations, marginals))
            real_pred1 = real_result >= 0.5
            real_accuracy = jnp.mean(real_pred1 * 1.0)
            real_pred_mean = jnp.mean(real_result)
            real_pred_max = jnp.max(real_result)
            real_pred_min = jnp.min(real_result)
            real_pred_std = jnp.std(real_result)
            real_pred_median = jnp.median(real_result)

            fake_result = jax.lax.stop_gradient(
                self.discriminator.apply(train_params['discriminator'], observations, latent_actions)
             ) 
            fake_pred1 = fake_result <= 0.5
            fake_accuracy = jnp.mean(fake_pred1 * 1.0)
            fake_pred_mean = jnp.mean(fake_result)
            fake_pred_max = jnp.max(fake_result)
            fake_pred_min = jnp.min(fake_result)
            fake_pred_std = jnp.std(fake_result)
            fake_pred_median = jnp.median(fake_result)


            """ RL Training """
            
            rng, split_rng = jax.random.split(rng)
            new_actions_rep, log_pi = self.policy.apply(train_params['policy'], split_rng, observations, deterministic=self.config.deterministic_action)
            new_actions = self.decoder.apply(train_params['decoder'], observations, new_actions_rep)


            if self.config.distance_logging:
                latent_dis_accuracy = self.discriminator.apply(train_params['discriminator'], observations, new_actions_rep).mean()
                bc_log_prob = self.bc_agent.log_likelihood(observations, new_actions)
                
                rng, split_rng = jax.random.split(rng)
                actions_rep, _ = self.encoder.apply(train_params['encoder'], split_rng, observations, actions)
                actions_rep = jnp.clip(actions_rep, -1.0 * self.latent_scale + self.config.epsilon, 1.0 * self.latent_scale - self.config.epsilon) 
                dataset_log_prob = -self.policy.apply(train_params['policy'], observations, actions_rep, method=self.policy.log_prob).mean()


            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (log_pi + self.config.target_entropy).mean()
                loss_collection['log_alpha'] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            _, q1_new_actions = self.qf.apply(train_params['qf1'], observations, new_actions)
            _, q2_new_actions = self.qf.apply(train_params['qf2'], observations, new_actions)
            q_new_actions = jnp.minimum(
                q1_new_actions, q2_new_actions
            )
            policy_loss = (alpha*log_pi - q_new_actions).mean()
            loss_collection['policy'] = policy_loss


            """ Q function loss """
            q1_pred_lastlayer, q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
            q2_pred_lastlayer, q2_pred = self.qf.apply(train_params['qf2'], observations, actions)
                

            q1_pred = jnp.clip(
                q1_pred,
                self.config.q_value_clip_min,
                self.config.q_value_clip_max,
            )
            q2_pred = jnp.clip(
                q2_pred,
                self.config.q_value_clip_min,
                self.config.q_value_clip_max,
            )

            rng, split_rng = jax.random.split(rng)
            if self.config.cql_max_target_backup:
                new_next_actions_rep, next_log_pi = self.policy.apply(
                    train_params['policy'], split_rng, next_observations, repeat=self.config.cql_n_actions
                )
                new_next_actions = self.decoder.apply(train_params['decoder'], next_observations, new_next_actions_rep)
                q1_next_pred_lastlayer, q1_next_actions = self.qf.apply(train_params['qf1'], next_observations, new_next_actions)
                q2_next_pred_lastlayer, q2_next_actions = self.qf.apply(train_params['qf2'], next_observations, new_next_actions)
                target_q_values = jnp.minimum(
                    q1_next_actions, q2_next_actions,
                ) 

                max_target_indices = jnp.expand_dims(jnp.argmax(target_q_values, axis=-1), axis=-1)
                target_q_values = jnp.take_along_axis(target_q_values, max_target_indices, axis=-1).squeeze(-1)
                next_log_pi = jnp.take_along_axis(next_log_pi, max_target_indices, axis=-1).squeeze(-1)
                    
            else:
                new_next_actions_rep, next_log_pi = self.policy.apply(
                    train_params['policy'], split_rng, next_observations, deterministic=self.config.deterministic_action,
                )
                new_next_actions = self.decoder.apply(train_params['decoder'], next_observations, new_next_actions_rep)

                q1_next_pred_lastlayer, q1_next_actions = self.qf.apply(train_params['qf1'], next_observations, new_next_actions)
                q2_next_pred_lastlayer, q2_next_actions = self.qf.apply(train_params['qf2'], next_observations, new_next_actions)
                target_q_values = jnp.minimum(
                    q1_next_actions, q2_next_actions,
                )
                
            # log decoder quality
            rng, split_rng = jax.random.split(rng) 
            encoded_new_next_action, _ = self.encoder.apply(train_params['encoder'], split_rng, next_observations, new_next_actions)
            next_action_l2 = mse_loss(encoded_new_next_action, new_next_actions_rep)

            target_q_values = jnp.clip(
                target_q_values,
                self.config.q_value_clip_min,
                self.config.q_value_clip_max,
            )
            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = jax.lax.stop_gradient(
                rewards + (1. - dones) * self.config.discount * target_q_values
            )

            qf1_loss = mse_loss(q1_pred, td_target)
            qf2_loss = mse_loss(q2_pred, td_target)

            ### CQL
            if self.config.use_cql:
                batch_size = actions.shape[0]
                rng, split_rng = jax.random.split(rng)
                cql_random_actions = jax.random.uniform(
                    split_rng, shape=(batch_size, self.config.cql_n_actions, self.action_dim),
                    minval=-1.0, maxval=1.0
                )

                rng, split_rng = jax.random.split(rng)
                cql_current_actions_rep, cql_current_log_pis = self.policy.apply(
                    train_params['policy'], split_rng, observations, repeat=self.config.cql_n_actions
                )
                cql_current_actions = self.decoder.apply(train_params['decoder'], observations, cql_current_actions_rep)

                rng, split_rng = jax.random.split(rng)
                cql_next_actions_rep, cql_next_log_pis = self.policy.apply(
                    train_params['policy'], split_rng, next_observations, repeat=self.config.cql_n_actions
                )
                cql_next_actions = self.decoder.apply(train_params['decoder'], next_observations, cql_next_actions_rep)

                _, cql_q1_rand = self.qf.apply(train_params['qf1'], observations, cql_random_actions)
                _, cql_q2_rand = self.qf.apply(train_params['qf2'], observations, cql_random_actions)
                _, cql_q1_current_actions = self.qf.apply(train_params['qf1'], observations, cql_current_actions)
                _, cql_q2_current_actions = self.qf.apply(train_params['qf2'], observations, cql_current_actions)
                _, cql_q1_next_actions = self.qf.apply(train_params['qf1'], observations, cql_next_actions)
                _, cql_q2_next_actions = self.qf.apply(train_params['qf2'], observations, cql_next_actions)

                cql_cat_q1 = jnp.concatenate(
                    [cql_q1_rand, jnp.expand_dims(q1_pred, 1), cql_q1_next_actions, cql_q1_current_actions], axis=1
                )
                cql_cat_q2 = jnp.concatenate(
                    [cql_q2_rand, jnp.expand_dims(q2_pred, 1), cql_q2_next_actions, cql_q2_current_actions], axis=1
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if self.config.cql_importance_sample:
                    random_density = np.log(0.5 ** self.latent_action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [cql_q1_rand - random_density,
                        cql_q1_next_actions - cql_next_log_pis,
                        cql_q1_current_actions - cql_current_log_pis],
                        axis=1
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [cql_q2_rand - random_density,
                        cql_q2_next_actions - cql_next_log_pis,
                        cql_q2_current_actions - cql_current_log_pis],
                        axis=1
                    )

                cql_qf1_ood = (
                    jax.scipy.special.logsumexp(cql_cat_q1 / self.config.cql_temp, axis=1)
                    * self.config.cql_temp
                )
                cql_qf2_ood = (
                    jax.scipy.special.logsumexp(cql_cat_q2 / self.config.cql_temp, axis=1)
                    * self.config.cql_temp
                )

                """Subtract the log likelihood of data"""
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()

                if self.config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(self.log_alpha_prime.apply(train_params['log_alpha_prime'])),
                        a_min=0.0, a_max=1000000.0
                    )
                    cql_min_qf1_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf1_diff - self.config.cql_target_action_gap)
                    cql_min_qf2_loss = alpha_prime * self.config.cql_min_q_weight * (cql_qf2_diff - self.config.cql_target_action_gap)

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss)*0.5

                    loss_collection['log_alpha_prime'] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_loss + cql_min_qf1_loss
                qf2_loss = qf2_loss + cql_min_qf2_loss

            loss_collection['qf1'] = qf1_loss
            loss_collection['qf2'] = qf2_loss

            if not rl:
                loss_collection['policy'] = 0.0
                loss_collection['qf1'] = 0.0
                loss_collection['qf2'] = 0.0
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }
        new_target_qf_params = {}
        new_target_qf_params['qf1'] = update_target_network(
            new_train_states['qf1'].params, target_qf_params['qf1'],
            self.config.soft_target_update_rate
        )
        new_target_qf_params['qf2'] = update_target_network(
            new_train_states['qf2'].params, target_qf_params['qf2'],
            self.config.soft_target_update_rate
        )

        metrics = dict(
            log_pi=aux_values['log_pi'].mean(),
            policy_loss=aux_values['policy_loss'],
            qf1_loss=aux_values['qf1_loss'],
            qf2_loss=aux_values['qf2_loss'],
            alpha_loss=aux_values['alpha_loss'],
            alpha=aux_values['alpha'],
            average_qf1=aux_values['q1_pred'].mean(),
            average_qf2=aux_values['q2_pred'].mean(),
            average_target_q=aux_values['target_q_values'].mean(),
            next_action_l2=aux_values['next_action_l2'].mean(),
        )

        metrics.update(prefix_metrics(dict(
            g_loss=aux_values['g_loss'],
            encoder_loss=aux_values['encoder_loss'],
            reconstruct_loss=aux_values['reconstruct_loss'],
            decoder_loss=aux_values['decoder_loss'],
            discriminator_loss=aux_values['d_loss'],  
            ), 'gan'))

        metrics.update(prefix_metrics(dict(
            real_accuracy=aux_values['real_accuracy'],
            fake_accuracy=aux_values['fake_accuracy'],
            real_pred_mean=aux_values['real_pred_mean'],
            fake_pred_mean=aux_values['fake_pred_mean'],
            real_pred_min=aux_values['real_pred_min'],
            fake_pred_min=aux_values['fake_pred_min'],
            real_pred_max=aux_values['real_pred_max'],
            fake_pred_max=aux_values['fake_pred_max'],
            real_pred_std=aux_values['real_pred_std'],
            fake_pred_std=aux_values['fake_pred_std'],
            real_pred_median=aux_values['real_pred_median'],
            fake_pred_median=aux_values['fake_pred_median'],    
            ), 'discriminator'))

        if self.config.use_cql:
            metrics.update(prefix_metrics(dict(
                cql_std_q1=aux_values['cql_std_q1'].mean(),
                cql_std_q2=aux_values['cql_std_q2'].mean(),
                cql_q1_rand=aux_values['cql_q1_rand'].mean(),
                cql_q2_rand=aux_values['cql_q2_rand'].mean(),
                cql_qf1_diff=aux_values['cql_qf1_diff'].mean(),
                cql_qf2_diff=aux_values['cql_qf2_diff'].mean(),
                cql_min_qf1_loss=aux_values['cql_min_qf1_loss'].mean(),
                cql_min_qf2_loss=aux_values['cql_min_qf2_loss'].mean(),
                cql_q1_current_actions=aux_values['cql_q1_current_actions'].mean(),
                cql_q2_current_actions=aux_values['cql_q2_current_actions'].mean(),
                cql_q1_next_actions=aux_values['cql_q1_next_actions'].mean(),
                cql_q2_next_actions=aux_values['cql_q2_next_actions'].mean(),
                alpha_prime=aux_values['alpha_prime'],
                alpha_prime_loss=aux_values['alpha_prime_loss'],
            ), 'cql'))
        
        if self.config.distance_logging:
            metrics.update(prefix_metrics(dict(
                latent_dis_accuracy=aux_values['latent_dis_accuracy'],
                bc_log_prob=aux_values['bc_log_prob'],
                dataset_log_prob=aux_values['dataset_log_prob'],
            ), 'cql'))

        return new_train_states, new_target_qf_params, metrics

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps
