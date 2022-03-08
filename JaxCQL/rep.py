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


class REP(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.encoder_lr = 3e-4
        config.dis_lr = 3e-4
        config.decoder_lr = 3e-4
        config.optimizer_type = 'adam'
        config.optimizer_b1 = 0.5
        config.optimizer_b2 = 0.999
        config.recon_alpha = 0.05
        config.gp_alpha = 10
        config.opt_n_critic = 1
        config.prior = 'gaussian'
        config.smooth_dis = False

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, encoder, discriminator, decoder, method='GAN'):
        self.config = self.get_default_config(config)
        self.encoder = encoder
        self.discriminator = discriminator
        self.decoder = decoder
        self.method = method
        self.observation_dim = encoder.observation_dim
        self.action_dim = encoder.action_dim
        self.latent_ac_dim = encoder.latent_action_dim

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        discriminator_params = self.discriminator.init(next_rng(), jnp.zeros((10, self.observation_dim)), jnp.zeros((10, self.latent_ac_dim)))
        self._train_states['discriminator'] = TrainState.create(
            params=discriminator_params,
            tx=optimizer_class(self.config.dis_lr, b1=self.config.optimizer_b1, b2=config.optimizer_b2),
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
            self._train_states, next_rng(), batch
        )
        return metrics

        
    @partial(jax.jit, static_argnames=('self'))
    def _train_step(self, train_states, rng, batch):

        def loss_fn(train_params, rng):
            observations = batch['observations']
            actions = batch['actions']
            batch_size, _ = jnp.shape(observations)
            adversarial_loss = mse_loss

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            latent_actions, _ = self.encoder.apply(train_params['encoder'], split_rng, observations, actions)

            if self.config.prior == 'uniform':
                rng, split_rng = jax.random.split(rng)
                marginals = jax.random.uniform(split_rng, (batch_size, self.latent_ac_dim), minval=-1.0, maxval=1.0)
            elif self.config.prior == 'gaussian':
                rng, split_rng = jax.random.split(rng)
                marginals = jax.random.multivariate_normal(split_rng, jnp.zeros(self.latent_ac_dim), jnp.diag(jnp.ones(self.latent_ac_dim)), (batch_size, ))
                
            valid = jnp.ones((batch_size))
            fake = jnp.zeros((batch_size))

            g_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, latent_actions), valid)
            reconstruct_loss = mse_loss(self.decoder.apply(train_params['decoder'], observations, latent_actions), actions)
            rep_loss = g_loss + self.config.recon_alpha * reconstruct_loss
            loss_collection['encoder'] = rep_loss

            loss_collection['decoder'] = reconstruct_loss
            # latent_actions, _ = self.encoder.apply(train_params['encoder'], split_rng, observations, actions)
            # reconstruct_loss_cp = mse_loss(self.decoder.apply(train_params['decoder'], observations, latent_actions), actions)
            # loss_collection['decoder'] = reconstruct_loss_cp

            if self.config.smooth_dis:
                rng, split_rng = jax.random.split(rng)
                bernoulli = jax.random.bernoulli(split_rng, 0.5, jnp.shape(valid)) * 0.2
                valid = valid - bernoulli

            real_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, marginals), valid)
            latent_actions_cp = jax.lax.stop_gradient(latent_actions)
            fake_loss = adversarial_loss(self.discriminator.apply(train_params['discriminator'], observations, latent_actions_cp), fake)
            d_loss = (real_loss + fake_loss) / 2
            loss_collection['discriminator'] = d_loss

            ### Accuracy ###
            real_pred1 = jax.lax.stop_gradient(
                self.discriminator.apply(train_params['discriminator'], observations, marginals)
             ) >= 0.5
            real_accuracy = jnp.mean(real_pred1 * 1.0)

            fake_pred1 = jax.lax.stop_gradient(
                self.discriminator.apply(train_params['discriminator'], observations, latent_actions_cp)
             ) <= 0.5
            fake_accuracy = jnp.mean(fake_pred1 * 1.0)

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            encoder_loss=aux_values['rep_loss'],
            decoder_loss=aux_values['reconstruct_loss'],
            discriminator_loss=aux_values['d_loss'],
            real_accuracy=aux_values['real_accuracy'],
            fake_accuracy=aux_values['fake_accuracy'],
        )

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
