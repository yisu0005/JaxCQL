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


class BC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.policy_lr = 3e-4
        config.optimizer_type = 'adam'
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.target_entropy = 0.0

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy):
        self.config = self.get_default_config(config)
        self.policy = policy

        self._train_states = {}

        optimizer_class = {
            'adam': optax.adam,
            'sgd': optax.sgd,
        }[self.config.optimizer_type]

        policy_params = self.policy.init(next_rng(), next_rng(), jnp.zeros((10, self.observation_dim)))
        self._train_states['policy'] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None
        )

        model_keys = ['policy']

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states['log_alpha'] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None
            )
            model_keys.append('log_alpha')

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

            loss_collection = {}

            rng, split_rng = jax.random.split(rng)
            new_actions, log_pi = self.policy.apply(train_params['policy'], split_rng, observations)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (log_pi + self.config.target_entropy).mean()
                loss_collection['log_alpha'] = alpha_loss
                alpha = jnp.exp(self.log_alpha.apply(train_params['log_alpha'])) * self.config.alpha_multiplier
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            rng, split_rng = jax.random.split(rng)
            log_probs = self.policy.apply(train_params['policy'], observations, actions, method=self.policy.log_prob)
            policy_loss = (alpha*log_pi - log_probs).mean()
            loss_collection['policy_loss'] = policy_loss
            negative_log_probs = -log_probs.mean()

            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(loss_fn, len(self.model_keys), has_aux=True)(train_params, rng)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }

        metrics = dict(
            policy_loss=aux_values['policy_loss'],
            negative_log_probs=-aux_values['negative_log_probs'],
            alpha_loss=aux_values['alpha_loss'],
            alpha=aux_values['alpha'],
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