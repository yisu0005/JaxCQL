from collections import OrderedDict
from copy import deepcopy
from functools import partial
import os
from ml_collections import ConfigDict
import cloudpickle as pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import distrax
import copy

from .jax_utils import next_rng, value_and_multi_grad, mse_loss
from .model import Scalar, update_target_network
from .utils import prefix_metrics


class REP(object):


    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.encoder_lr = 3e-4
        config.dis_lr = 3e-4
        config.decoder_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = 'adam'
        config.optimizer_b1 = 0.5 
        config.optimizer_b2 = 0.999
        config.recon_alpha = 0.01
        config.z_alpha = 0.1
        config.decoder_z_alpha = 0.0
        config.qf_alpha = 0.0
        config.prior = 'uniform'
        config.smooth_dis = False
        config.smooth_decoder = False
        config.latent_stats_logging = True
        config.sarsa = False
        config.discount = 0.99
        config.use_automatic_qf_tuning = False
        config.soft_target_update_rate = 5e-3

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, encoder, discriminator, decoder, qf, method='GAN'):
        self.config = self.get_default_config(config)
        self.encoder = encoder
        self.discriminator = discriminator
        self.decoder = decoder
        self.method = method
        self.observation_dim = encoder.observation_dim
        self.action_dim = encoder.action_dim
        self.latent_ac_dim = encoder.latent_action_dim
        self.dropout = discriminator.dropout
        self.qf = qf

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        discriminator_params = self.discriminator.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.latent_ac_dim)))
        self._train_states['discriminator'] = TrainState.create(
            params=discriminator_params,
            tx=optimizer_class(self.config.dis_lr, b1=self.config.optimizer_b1, b2=self.config.optimizer_b2),
            apply_fn=None,
        )

        decoder_params = self.decoder.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.latent_ac_dim)))
        self._train_states['decoder'] = TrainState.create(
            params=decoder_params,
            tx=optimizer_class(self.config.decoder_lr),
            apply_fn=None,
        )

        encoder_params = self.encoder.init(next_rng(), next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.action_dim)))
        self._train_states['encoder'] = TrainState.create(
            params=encoder_params,
            tx=optimizer_class(self.config.encoder_lr),
            apply_fn=None,
        )

        model_keys = ['discriminator', 'encoder', 'decoder']

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_step(
            self._train_states, next_rng(), batch, True
        )
        return metrics

    def val(self, batch):
        _, metrics = self._train_step(
            self._train_states, next_rng(), batch, False
        )
        return metrics

        
    @partial(jax.jit, static_argnames=('self', 'train'))
    def _train_step(self, train_states, rng, batch, train=True):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            actions = batch['actions']
            next_observations = batch['next_observations']
            rewards = batch['rewards']
            dones = batch['dones']


            batch_size, _ = jnp.shape(observations)
            adversarial_loss = mse_loss

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            (latent_actions, _) = self.encoder.apply(train_params['encoder'], split_rng, observations, actions)

            if self.config.prior == 'uniform':
                rng, split_rng = jax.random.split(rng)
                marginals = jax.random.uniform(split_rng, (batch_size, self.latent_ac_dim), minval=-1.0, maxval=1.0)
            elif self.config.prior == 'gaussian':
                rng, split_rng = jax.random.split(rng)
                marginals = jax.random.multivariate_normal(split_rng, jnp.zeros(self.latent_ac_dim), jnp.diag(jnp.ones(self.latent_ac_dim)), (batch_size, ))
                
            valid = jnp.ones((batch_size))
            fake = jnp.zeros((batch_size))

            g_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, latent_actions), valid)
            decoded_actions = self.decoder.apply(train_params['decoder'], observations, latent_actions)
            reconstruct_loss = mse_loss(decoded_actions, actions)


            rep_loss = g_loss + self.config.recon_alpha * reconstruct_loss


            if self.config.prior == 'uniform':
                rng, split_rng = jax.random.split(rng)
                random_z = jax.random.uniform(split_rng, (batch_size, self.latent_ac_dim), minval=-1.0, maxval=1.0)
            elif self.config.prior == 'gaussian':
                rng, split_rng = jax.random.split(rng)
                random_z = jax.random.multivariate_normal(split_rng, jnp.zeros(self.latent_ac_dim), jnp.diag(jnp.ones(self.latent_ac_dim)), (batch_size, ))

            random_z_decoded = self.decoder.apply(train_params['decoder'], observations, random_z)
            rng, split_rng = jax.random.split(rng)
            (random_z_reconstructed, _) = self.encoder.apply(train_params['encoder'], split_rng, observations, random_z_decoded)
            random_z_distance = mse_loss(random_z, random_z_reconstructed)

            rng, split_rng = jax.random.split(rng)
            noise = jax.random.uniform(split_rng, (batch_size, self.action_dim), minval=-1.0, maxval=1.0)
            random_a = actions + noise * 0.1
            rng, split_rng = jax.random.split(rng)
            (random_a_rep, _) = self.encoder.apply(train_params['encoder'], split_rng, observations, random_a)
            random_a_reconstructed = self.decoder.apply(train_params['decoder'], observations, random_a_rep)
            random_a_distance = mse_loss(actions, random_a_reconstructed)

            encoder_loss = rep_loss + self.config.z_alpha * random_z_distance + self.config.z_alpha * random_a_distance

            loss_collection['encoder'] = encoder_loss 

            if self.config.smooth_decoder:
                rng, split_rng = jax.random.split(rng)
                noise = jax.random.normal(split_rng, jnp.shape(latent_actions)) * 0.2
                perturbed_latent_actions = latent_actions + noise
                perturbed_reconstruct_loss = mse_loss(self.decoder.apply(train_params['decoder'], observations, perturbed_latent_actions), actions)
            else:
                perturbed_reconstruct_loss = reconstruct_loss
            
            decoder_loss = perturbed_reconstruct_loss + self.config.decoder_z_alpha * random_z_distance + self.config.decoder_z_alpha * random_a_distance 
            loss_collection['decoder'] = decoder_loss

            if self.config.smooth_dis:
                rng, split_rng = jax.random.split(rng)
                bernoulli = jax.random.bernoulli(split_rng, 0.5, jnp.shape(valid)) * 0.2
                valid = valid - bernoulli

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

            ### latent action statistics ###
            if self.config.latent_stats_logging:
                _ , actions_mean, actions_logstd = self.encoder.apply(train_params['encoder'], split_rng, observations, actions, method=self.encoder.get_statistics)
                ac_std = jnp.exp(actions_logstd)
                latent_ac_max = jnp.max(actions_mean)
                latent_ac_min = jnp.min(actions_mean)
                latent_ac_mean = jnp.mean(actions_mean)
                latent_ac_std = jnp.std(actions_mean)

                latent_ac_std_max = jnp.max(ac_std)
                latent_ac_std_min = jnp.min(ac_std)
                latent_ac_std_mean = jnp.mean(ac_std)
                latent_ac_std_std = jnp.std(ac_std) 

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        if train:
            new_train_states = {
                key: train_states[key].apply_gradients(grads=grads[i][key])
                for i, key in enumerate(self.model_keys)
            }
        else:
            new_train_states = copy.deepcopy(train_states)

        metrics = dict(
            g_loss=aux_values['g_loss'],
            encoder_loss=aux_values['encoder_loss'],
            reconstruct_loss=aux_values['reconstruct_loss'],
            decoder_loss=aux_values['decoder_loss'],
            random_z_distance=aux_values['random_z_distance'],
            random_a_distance=aux_values['random_a_distance'],
            perturbed_decoder_loss=aux_values['perturbed_reconstruct_loss'],
            discriminator_loss=aux_values['d_loss'],
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
        )

        if self.config.latent_stats_logging:
            metrics.update(prefix_metrics(dict(
                latent_ac_max=aux_values['latent_ac_max'],
                latent_ac_min=aux_values['latent_ac_min'],
                latent_ac_mean=aux_values['latent_ac_mean'],
                latent_ac_std=aux_values['latent_ac_std'],

                latent_ac_std_max=aux_values['latent_ac_std_max'],
                latent_ac_std_min=aux_values['latent_ac_std_min'],
                latent_ac_std_mean=aux_values['latent_ac_std_mean'],
                latent_ac_std_std=aux_values['latent_ac_std_std'],

            ), 'latent_actions'))
        return new_train_states, metrics



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

